"""Checks, on a CPU, that the notebook scripts do what the recorded runs did (the Colab scripts of the retrieval code base).
All inputs are paths given as arguments; nothing in them is written to.

    V1  script/evaluate.py rank on the recorded test outputs gives the recorded ranks, Recall@K and category accuracy
    V2  choose_cfg on the recorded validation tables gives the recorded guidance weights
    V3  add_categories gives the recorded samples.jsonl byte for byte
    V4  the commands of train_run.py and eval_run.py (--print) equal the recorded command lines
    V5  script.data.load_rooms gives the same rooms and item categories as the loader of the retrieval code base

    python notebooks/verify_port.py --runs_dir <recorded runs> --kit <room files> --val100 <rooms_val100.json>
        --gallery <clip_vectors> --retrieval_root <checkout of the retrieval code base at 2aa1dda> --scratch <empty dir>
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from hd_common import RUNS, add_categories, choose_cfg  # noqa: E402
from script.evaluate import load_gallery, rank_results  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--runs_dir", required=True, help="the recorded runs (runs/<run>, runs/eval_<run>)")
p.add_argument("--kit", required=True, help="directory with rooms_train.json, rooms_val.json, rooms_test0.json")
p.add_argument("--val100", required=True, help="rooms_val100.json")
p.add_argument("--gallery", required=True, help="directory of the clip test gallery pkl files")
p.add_argument("--retrieval_root", default="", help="checkout of the retrieval code base at 2aa1dda (for V5)")
p.add_argument("--scratch", required=True)
p.add_argument("--only", default="V1,V2,V3,V4,V5")
args = p.parse_args()
only = set(args.only.split(","))
os.makedirs(args.scratch, exist_ok=True)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


test_dirs = sorted(glob.glob(os.path.join(args.runs_dir, "eval_dit_*", "test0_members*")))

if "V1" in only:
    print(f"V1 rank on {len(test_dirs)} recorded test outputs", flush=True)
    gallery = load_gallery(args.gallery, "clip_vectors")
    for d in test_dirs:
        name = os.path.relpath(d, args.runs_dir)
        out = os.path.join(args.scratch, "V1", name)
        summary = rank_results(d, gallery, out)
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
            recorded = json.load(f)
        keys = ["count", "recall_clip@1", "recall_clip@5", "recall_clip@10", "recall_clip@20", "category_correct"]
        same_summary = all(summary[k] == recorded[k] for k in keys) and set(summary) == set(keys)
        with open(os.path.join(d, "per_sample_results.jsonl"), encoding="utf-8") as f:
            rec_rows = [{k: v for k, v in json.loads(line).items() if k != "rank_style"} for line in f]
        with open(os.path.join(out, "per_sample_results.jsonl"), encoding="utf-8") as f:
            new_rows = [json.loads(line) for line in f]
        same_rows = rec_rows == new_rows
        same_cat = open(os.path.join(d, "summary_category.csv"), "rb").read() == open(os.path.join(out, "summary_category.csv"), "rb").read()
        with open(os.path.join(d, "summary_topk.csv"), newline="", encoding="utf-8") as f:
            rec_topk = [r for r in csv.reader(f) if r[0] in ("metric", "clip")]
        with open(os.path.join(out, "summary_topk.csv"), newline="", encoding="utf-8") as f:
            new_topk = list(csv.reader(f))
        check(f"V1 {name}", same_summary and same_rows and same_cat and rec_topk == new_topk,
              f"summary {same_summary}, rows {same_rows} ({len(new_rows)}), category csv {same_cat}, topk {rec_topk == new_topk}; "
              f"R@1 {summary['recall_clip@1']:.2f}")

if "V2" in only:
    print("V2 choice of the guidance weight", flush=True)
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "eval_dit_*", "eval_record.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("cfg_fixed_for_reference"):
            continue
        table = rec["cfg_selection_on_val100"][f"members{rec['query_members_trained']}"]
        got = choose_cfg(rec["cfg_candidates"], table)
        check(f"V2 {rec['run']}", got == rec["cfg_selected"], f"chosen {got}, recorded {rec['cfg_selected']}, "
              f"R@1 {[round(table[str(c)]['recall_clip@1'], 2) for c in rec['cfg_candidates']]}")

if "V3" in only:
    print("V3 categories added to samples.jsonl", flush=True)
    for d in test_dirs:
        name = os.path.relpath(d, args.runs_dir)
        out = os.path.join(args.scratch, "V3", name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(d, "samples.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        with open(os.path.join(out, "samples.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps({k: v for k, v in r.items() if k != "category"}) + "\n" for r in rows)
        n = add_categories(out, os.path.join(args.kit, "rooms_test0.json"))
        same = open(os.path.join(d, "samples.jsonl"), "rb").read() == open(os.path.join(out, "samples.jsonl"), "rb").read()
        check(f"V3 {name}", same, f"{n} samples")

if "V4" in only:
    print("V4 commands", flush=True)

    def printed(script, *extra):
        res = subprocess.run([sys.executable, os.path.join(HERE, script), *extra, "--print"], capture_output=True, text=True, check=True)
        return res.stdout.splitlines()

    for run in sorted(RUNS):
        for mode, log in (("full", "train.log"), ("smoke", "smoke.log")):
            path = os.path.join(args.runs_dir, run, log)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                recorded = {line.rstrip("\n").split(f"[vm_train_dit] {run} {mode} ", 1)[1]
                            for line in f if line.startswith(f"[vm_train_dit] {run} {mode} ")}
            if not recorded:
                continue
            line = printed("train_run.py", "--run", run, *(["--smoke"] if mode == "smoke" else []))[0]
            ours = line.split(f"[train_run] {run} {mode} ", 1)[1]
            check(f"V4 train {run} {mode}", recorded == {ours}, f"{len(recorded)} distinct recorded line(s)")

    module = {"script.generate": "script.generate", "methods.heterogeneous_diffusion.evaluate": "script.evaluate"}

    def normalized(line):
        """'$ <python> -m <module> ...' -> (module, args), with the evaluation arguments of the clip-only port."""
        m = re.match(r"\$ \S+ -m (\S+) (.*)$", line.strip())
        if not m:
            return None
        mod, rest = m.group(1), m.group(2)
        rest = rest.replace(" --style 0", "").replace(" --style_gallery  ", " ")
        return module.get(mod, mod), rest

    for path in sorted(glob.glob(os.path.join(args.runs_dir, "eval_dit_*", "train.log"))):
        entry = os.path.basename(os.path.dirname(path))
        run, _, tag = entry[len("eval_"):].partition("@")
        with open(path, encoding="utf-8", errors="replace") as f:
            recorded = [normalized(line) for line in f if line.startswith("$ ") and " -m " in line]
        recorded = [r for r in recorded if r and r[0] in ("script.generate", "script.evaluate")]
        rec_set = set(recorded)
        # test split 0 with the weight that the run used, and (without a fixed weight) the validation rooms
        er_path = glob.glob(os.path.join(args.runs_dir, entry, "eval_record.json"))
        if not er_path:
            continue
        with open(er_path[0], encoding="utf-8") as f:
            cfg = json.load(f)["cfg_selected"]
        ours = [normalized(x) for x in printed("eval_run.py", "--run", run, "--cfg", str(cfg))]
        if not tag:  # the validation commands of the choice; the test outputs then live in /content/eval/<run>, not ..._cfgX
            ours = [(m, a.replace(f"/content/eval/{run}_cfg{cfg}/", f"/content/eval/{run}/")) for m, a in ours]
            ours += [normalized(x) for x in printed("eval_run.py", "--run", run)]
        gen_ours = {o for o in ours if o[0] == "script.generate"}
        gen_rec = {r for r in rec_set if r[0] == "script.generate"}
        check(f"V4 generate {entry}", gen_ours == gen_rec, f"{len(gen_rec)} recorded, {len(gen_ours)} ours")
        ev_ours = {o for o in ours if o[0] == "script.evaluate" and "--clip_gallery" not in o[1]}
        ev_rec = {r for r in rec_set if r[0] == "script.evaluate"}
        check(f"V4 evaluate {entry}", ev_ours == ev_rec, f"{len(ev_rec)} recorded, {len(ev_ours)} ours "
              "(the test-gallery rank ran on the server before, so it has no recorded line)")

if "V5" in only and args.retrieval_root:
    print("V5 room loader", flush=True)
    from script.data import load_rooms as ours_load

    spec = importlib.util.spec_from_file_location("retrieval_hd_data", os.path.join(args.retrieval_root, "methods", "heterogeneous_diffusion", "data.py"))
    theirs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = theirs  # dataclasses look the module up while the class is created
    sys.path.insert(0, args.retrieval_root)
    spec.loader.exec_module(theirs)
    for name in ("rooms_train.json", "rooms_val.json", "rooms_test0.json"):
        a = [vars(r) for r in ours_load(os.path.join(args.kit, name))]
        b = [vars(r) for r in theirs.load_rooms(os.path.join(args.kit, name))]
        check(f"V5 rooms {name}", a == b, f"{len(a)} rooms")
    a = [vars(r) for r in ours_load(args.val100)]
    b = [vars(r) for r in theirs.load_rooms(args.val100)]
    check("V5 rooms rooms_val100.json", a == b, f"{len(a)} rooms")

    def category_of(load):
        out = {}
        for name in ("rooms_train.json", "rooms_val.json"):
            for room in load(os.path.join(args.kit, name)):
                out.update(zip(room.query_ids + room.target_ids, room.query_categories + room.target_categories))
        return out

    a, b = category_of(ours_load), category_of(theirs.load_rooms)
    check("V5 item categories (train then val)", a == b, f"{len(a)} items")

shutil.rmtree(os.path.join(args.scratch, "V3"), ignore_errors=True)
print(f"\n{'all checks passed' if not failures else 'FAILED: ' + ', '.join(failures)}", flush=True)
raise SystemExit(1 if failures else 0)
