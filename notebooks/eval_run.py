"""Evaluation of one trained DiT run (one blocking cell). Safe to run again; finished steps are skipped.

    1) Choose the guidance weight on the 100 validation rooms from 1.5 / 2.5 / 3.5 / 5.0 by clip Recall@1%
       (gallery = items of the training and validation rooms, per category). One weight per model; a model trained
       with the query images in the groups chooses with them (the table without them is kept as well). Ties go to
       the smaller weight.
    2) Generate test split 0 with the chosen weight (models trained with the query images: with and without them).
       256 px, 250 steps, EMA weights.
    3) Clip embeddings, then ranks, Recall@K and category accuracy against the test gallery (clip only).

    python notebooks/eval_run.py --run dit_embed_f                # choose the weight, then test split 0
    python notebooks/eval_run.py --run dit_embed_f --cfg 5.0      # fixed weight (reproduces the recorded table)
    python notebooks/eval_run.py --run dit_embed_f --cfg 5.0 --print
    python notebooks/eval_run.py --run dit_embed_f --upload       # also put the results on the branch eval-<run>

A fixed weight writes to /content/eval/<run>_cfg<weight> and the branch eval-<run>-cfg<weight>. An existing eval branch
is never written to.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hd_common import CACHE, CODE, EVAL_DIR, HF_REPO, KIT, KIT_EVAL, RUNS, RUNS_DIR, add_categories, choose_cfg  # noqa: E402

MEASURES = ("count", "recall_clip@1", "recall_clip@5", "recall_clip@10", "recall_clip@20", "category_correct")

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, choices=sorted(RUNS))
p.add_argument("--cfg", default=None, type=float, help="fixed guidance weight; no choice on the validation rooms")
p.add_argument("--cfgs", default="1.5,2.5,3.5,5.0", help="candidates of the choice")
p.add_argument("--hf_repo", default=HF_REPO)
p.add_argument("--gen_extra", default="", help="extra arguments of script.generate")
p.add_argument("--upload", action="store_true", help="put the results on the branch eval-<run> (only if it does not exist)")
p.add_argument("--print", action="store_true", help="only print the commands (test split 0 needs --cfg)")
args = p.parse_args()

run = args.run
form, members, embedding = RUNS[run]
TAG = f"cfg{args.cfg}" if args.cfg is not None else ""
CFGS = [args.cfg] if TAG else [float(c) for c in args.cfgs.split(",")]
GEN_EXTRA = shlex.split(args.gen_extra)
EVAL = f"{EVAL_DIR}/{run}" + (f"_{TAG}" if TAG else "")
REC = f"{RUNS_DIR}/eval_{run}" + (f"@{TAG}" if TAG else "")
ckpt = f"{RUNS_DIR}/{run}/final_ema.pt"


def gen_cmd(rooms_path, cfg, with_members, out):
    return [sys.executable, "-m", "script.generate", "--ckpt", ckpt, "--rooms", rooms_path, "--cache_dir", CACHE, "--cfg", str(cfg),
            "--out", out, "--query_members", str(with_members), "--batch_size", "64"] + GEN_EXTRA


def embed_cmd(out):
    return [sys.executable, "-m", "script.evaluate", "embed", "--result", out]


def item_rank_cmd(out):
    return [sys.executable, "-m", "script.evaluate", "rank", "--result", out, "--item_gallery", CACHE,
            "--item_gallery_rooms", f"{KIT}/rooms_train.json", f"{KIT}/rooms_val.json"]


def test_rank_cmd(out):
    return [sys.executable, "-m", "script.evaluate", "rank", "--result", out, "--clip_gallery", f"{KIT}/clip_vectors"]


if args.print:  # in the order of execution
    for m in ([] if TAG else [members, 0] if members else [members]):
        for cfg in CFGS:
            out = f"{EVAL}/val100_members{m}_cfg{cfg}"
            for cmd in (gen_cmd(f"{KIT_EVAL}/rooms_val100.json", cfg, m, out), embed_cmd(out), item_rank_cmd(out)):
                print("$ " + " ".join(cmd))
    if TAG:
        for m in ([0, 1] if members else [0]):
            out = f"{EVAL}/test0_members{m}"
            for cmd in (gen_cmd(f"{KIT}/rooms_test0.json", args.cfg, m, out), embed_cmd(out), test_rank_cmd(out)):
                print("$ " + " ".join(cmd))
    raise SystemExit(0)

t0 = time.time()
for d in (KIT_EVAL, EVAL, REC):
    os.makedirs(d, exist_ok=True)
LOG = open(f"{REC}/train.log", "a", encoding="utf-8")


def say(text):
    print(text, flush=True)
    LOG.write(text + "\n"); LOG.flush()


def record(name, **values):  # records of each step
    with open(f"{REC}/{name}", "a", encoding="utf-8") as f:
        f.write(json.dumps({"elapsed_min": round((time.time() - t0) / 60, 1), **values}) + "\n")


def sh(cmd, cwd=CODE):
    say(f"$ {' '.join(cmd)[:260]}")
    t1 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = []
    for line in p.stdout:  # stream the output as it comes
        line = line.rstrip()
        if line:
            print(line[:300], flush=True)
            last = (last + [line])[-15:]
    rc = p.wait()
    say(f"  ({time.time() - t1:.0f} s, exit {rc})")
    if rc:
        LOG.write("\n".join(last) + "\n"); LOG.flush()
        raise RuntimeError(f"exit {rc}: {' '.join(cmd)[:200]}")


say(f"[eval_run] run={run} form={form} query_members={members} query_embedding={embedding} cfgs={CFGS}")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

branch = f"eval-{run}" + (f"-{TAG}" if TAG else "")
api = HfApi()
if args.upload and any(b.name == branch for b in api.list_repo_refs(args.hf_repo).branches):
    raise SystemExit(f"[eval_run] {args.hf_repo}@{branch} exists; not writing to it")

import torch  # noqa: E402

if not os.path.isfile(ckpt):  # on another VM than the training: take the weights from the branch
    hf_hub_download(args.hf_repo, "final_ema.pt", revision=f"run-{run}", local_dir=os.path.dirname(ckpt))
weights_step = int(torch.load(ckpt, map_location="cpu")["global_step"])
say(f"[eval_run] weights {ckpt} step={weights_step}")
with open(f"{REC}/args.json", "w", encoding="utf-8") as f:
    json.dump({"run": run, "form": form, "query_members_trained": members, "query_embedding": embedding, "weights_step": weights_step,
               "cfg_candidates": CFGS, "generate_extra": GEN_EXTRA}, f, indent=1)


def generate(rooms_path, cfg, with_members, out):
    if not os.path.isfile(f"{out}/generate_args.json"):
        sh(gen_cmd(rooms_path, cfg, with_members, out))
    return add_categories(out, rooms_path)


def embed(out):
    if not os.path.isfile(f"{out}/clip_embeds.npy"):
        sh(embed_cmd(out))


def select(with_members):
    table = {}
    for cfg in CFGS:
        out = f"{EVAL}/val100_members{with_members}_cfg{cfg}"
        generate(f"{KIT_EVAL}/rooms_val100.json", cfg, with_members, out)
        embed(out)
        if not os.path.isfile(f"{out}/summary.json"):
            sh(item_rank_cmd(out))
        with open(f"{out}/summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        table[str(cfg)] = {k: summary[k] for k in MEASURES}
        say(f"[cfg {cfg} members={with_members}] {json.dumps(table[str(cfg)])}")
        record("val_loss.jsonl", stage="cfg_selection", query_members=with_members, cfg=cfg, **table[str(cfg)])
    return table


# 1) guidance weight
if TAG:  # fixed weight: no choice
    selection, used, best = {}, {}, CFGS[0]
    say(f"[eval_run] fixed cfg={best} (no choice)")
else:
    selection = {f"members{members}": select(members)}
    used = selection[f"members{members}"]
    best = choose_cfg(CFGS, used)
    say(f"[eval_run] selected cfg={best} (chosen with query_members={members})")
    if members:  # for reference: the table without the query images
        selection["members0"] = select(0)

# 2) and 3) test split 0: generation, clip embeddings, ranks against the test gallery
result = {"run": run, "form": form, "query_members_trained": members, "query_embedding": embedding, "weights_step": weights_step,
          "cfg_candidates": CFGS, "cfg_selection_on_val100": selection, "cfg_selected": best, "cfg_selected_with_query_members": members,
          "cfg_fixed_for_reference": bool(TAG),
          "steps": 250, "test": {}}
files = ["samples.jsonl", "clip_embeds.npy", "generate_args.json", "images.tar",
         "per_sample_results.jsonl", "summary.json", "summary_topk.csv", "summary_category.csv"]
for m in ([0, 1] if members else [0]):
    name = f"test0_members{m}"; out = f"{EVAL}/{name}"
    count = generate(f"{KIT}/rooms_test0.json", best, m, out)
    embed(out)
    if not os.path.isfile(f"{out}/summary.json"):
        sh(test_rank_cmd(out))
    with open(f"{out}/summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    if not os.path.isfile(f"{out}/images.tar"):
        with tarfile.open(f"{out}/images.tar.tmp", "w") as tar:  # the images as one file (PNG, so no compression)
            tar.add(f"{out}/images", arcname="images")
        os.replace(f"{out}/images.tar.tmp", f"{out}/images.tar")
    result["test"][name] = {"targets": count, "images_tar_mb": round(os.path.getsize(f"{out}/images.tar") / 2**20, 1),
                            **{k: summary[k] for k in MEASURES}}
    say(f"[eval_run] {name}: {json.dumps(result['test'][name])}")
    record("log.jsonl", stage="test", condition=name, cfg=best, **result["test"][name])
with open(f"{EVAL}/eval_record.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
if args.upload:
    for attempt in range(8):
        try:
            api.create_branch(args.hf_repo, branch=branch, exist_ok=False)
            break
        except Exception as e:
            say(f"[eval_run] branch creation failed (attempt {attempt + 1}): {repr(e)[:300]}")
            time.sleep(120)
    else:
        raise RuntimeError("cannot create the eval branch")
    for attempt in range(8):
        try:
            api.upload_folder(folder_path=EVAL, repo_id=args.hf_repo, revision=branch,
                              allow_patterns=["eval_record.json"] + [f"{c}/{f}" for c in result["test"] for f in files])
            say(f"[eval_run] uploaded to {args.hf_repo}@{branch}")
            break
        except Exception as e:
            say(f"[eval_run] upload failed (attempt {attempt + 1}): {repr(e)[:300]}")
            time.sleep(120)
    else:
        raise RuntimeError("cannot upload the results")
record("val_loss.jsonl", stage="fixed" if TAG else "selected", cfg=best, conditions=sorted(result["test"]), **used.get(str(best), {}))
say(f"[eval_run] run={run} ok=1 cfg={best} conditions={','.join(sorted(result['test']))} elapsed_min={(time.time() - t0) / 60:.1f}")
LOG.close()
