"""Training of one DiT run, to the end or until the connection is lost (one blocking cell). Safe to run again: --resume
continues exactly from the latest checkpoint on the VM or on the Hugging Face branch.

    python notebooks/train_run.py --run dit_embed_f --smoke   # 20 updates; the smoke branch and outputs are removed
    python notebooks/train_run.py --run dit_embed_f           # the run; checkpoints go to the branch run-<run>
    python notebooks/train_run.py --run dit_embed_f --print   # only print the command

The arguments of script.train are the ones of the recorded runs. A branch that already holds final_ema.pt is not trained
again: the recorded branches no longer hold latest.pt, so --resume would start from scratch and replace the recorded
weights. Give another --hf_branch to train such a run again.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hd_common import CACHE, CODE, HF_REPO, KIT, RUNS, RUNS_DIR, WEIGHTS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, choices=sorted(RUNS))
p.add_argument("--smoke", action="store_true")
p.add_argument("--hf_repo", default=HF_REPO)
p.add_argument("--hf_branch", default="", help="defaults to run-<run> (smoke-<run> with --smoke)")
p.add_argument("--chunk_images", default="0", help="images per forward pass (0 = no limit)")
p.add_argument("--epochs", default="100")
p.add_argument("--print", action="store_true", help="only print the command")
args = p.parse_args()

run = args.run
form, members, embedding = RUNS[run]
smoke = args.smoke
branch = args.hf_branch or (f"smoke-{run}" if smoke else f"run-{run}")
out = f"{RUNS_DIR}/{run}" + ("_smoke" if smoke else "")
cmd = [sys.executable, "-m", "script.train", "--form", form, "--query_members", str(members), "--query_embedding", str(embedding),
       "--rooms_train", f"{KIT}/rooms_train.json", "--rooms_val", f"{KIT}/rooms_val.json", "--cache_dir", CACHE,
       "--init_ckpt", WEIGHTS, "--out", out, "--chunk_images", args.chunk_images,
       "--epochs", args.epochs, "--log_every", "20"]
if smoke:  # the path of the run, shortened: validation loss, checkpoint saving, upload to the repository
    cmd += ["--max_steps", "20", "--log_every", "5", "--limit_rooms", "400", "--val_every", "1", "--save_minutes", "0.5",
            "--hf_repo", args.hf_repo, "--hf_branch", branch]
else:
    cmd += ["--resume", "--hf_repo", args.hf_repo, "--hf_branch", branch]
line = f"[train_run] {run} {'smoke' if smoke else 'full'} {' '.join(cmd[2:])}"
if args.print:
    print(line)
    raise SystemExit(0)

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.errors import RevisionNotFoundError  # noqa: E402

api = HfApi()
if not smoke:
    try:
        on_hub = set(api.list_repo_files(args.hf_repo, revision=branch))
    except RevisionNotFoundError:  # the branch does not exist yet; any other failure stops here
        on_hub = set()
    if "final_ema.pt" in on_hub:
        raise SystemExit(f"[train_run] {args.hf_repo}@{branch} already holds final_ema.pt; not training it again "
                         "(give another --hf_branch to train this run again)")
print(line, flush=True)
t0 = time.time()
if smoke:
    subprocess.run(["rm", "-rf", out])
proc = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for text in proc.stdout:
    print(text.rstrip(), flush=True)
rc = proc.wait()
finished = os.path.isfile(f"{out}/final_ema.pt")
if smoke:  # the smoke branch and outputs are not kept
    try:
        files = api.list_repo_files(args.hf_repo, revision=branch)
        print("[train_run] smoke branch files:", sorted(files), flush=True)
        api.delete_branch(args.hf_repo, branch=branch)
    except Exception as e:
        print("[train_run] smoke branch cleanup failed:", repr(e)[:200], flush=True)
    val = os.path.isfile(f"{out}/val_loss.jsonl") and open(f"{out}/val_loss.jsonl").read().strip().splitlines()
    print("[train_run] smoke val_loss lines:", len(val or []), (val or [""])[-1][:200], flush=True)
    subprocess.run(["rm", "-rf", out])
print(f"[train_run] run={run} smoke={int(smoke)} exit {rc} finished={int(finished)} elapsed_min={(time.time() - t0) / 60:.1f}", flush=True)
raise SystemExit(rc)
