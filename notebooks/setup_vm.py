"""Preparation of the Colab VM for the DiT runs (no Drive). Safe to run again; finished steps are skipped.

Run from the notebook after the code has been cloned at the pinned commit:
    python /content/GroupDiff/notebooks/setup_vm.py

    1) the Hugging Face token (written by the login cell) and the private repository
    2) dependencies (notebooks/requirements-colab.txt) and the CPU checks of script/check.py
    3) room files, the image list and the test gallery from the data branch, checked against hf_files.sha256
    4) the feature cache from the cache branch, checked the same way
       (--recompute_cache computes it from the item images instead; --upload_cache also puts it on the branch)
    5) the public DiT-XL/2 weights (CC BY-NC 4.0), checked the same way
    6) a record of the session (/content/runs/colab_session.json)
"""
import argparse
import datetime
import glob
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hd_common import (CACHE, CACHE_BRANCH, CACHE_FILES, CODE, DATA_BRANCH, DATA_FILES, FURN, HF_REPO, KIT, RUNS_DIR,  # noqa: E402
                       WEIGHTS, WEIGHTS_URL, check_file, expected_sha256, fetch, sh)

# versions of the recorded runs that the Colab image decides (printed and compared, not installed)
RECORDED_IMAGE = {"python": "3.13.15", "torch": "2.11.0+cu130"}

p = argparse.ArgumentParser()
p.add_argument("--hf_repo", default=HF_REPO)
p.add_argument("--data_branch", default=DATA_BRANCH)
p.add_argument("--recompute_cache", action="store_true", help="compute the feature cache from the item images")
p.add_argument("--upload_cache", action="store_true", help="with --recompute_cache: put the new cache on the cache branch")
args = p.parse_args()
expected = expected_sha256()

# 1) token and repository
token_path = os.path.expanduser("~/.cache/huggingface/token")
assert os.path.isfile(token_path), "no Hugging Face token: run the login cell first"
os.chmod(token_path, 0o600)
rev = subprocess.run(["git", "-C", CODE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"[code] {CODE} at {rev}", flush=True)

# 2) dependencies (torch and Python are the Colab image's) and the CPU checks
req = os.path.join(CODE, "notebooks", "requirements-colab.txt")
sh(f"pip install -q -r {req} 2>&1 | tail -3; "
   "python -c \"import torch, timm, einops, diffusers, transformers, huggingface_hub, torchdiffeq; "
   "print('torch', torch.__version__, 'timm', timm.__version__, 'diffusers', diffusers.__version__, "
   "'transformers', transformers.__version__, 'huggingface_hub', huggingface_hub.__version__)\"")
import torch  # noqa: E402

image = {"python": platform.python_version(), "torch": torch.__version__}
for k, v in RECORDED_IMAGE.items():
    if image[k] != v:
        print(f"[warn] {k} {image[k]} differs from the recorded runs ({v})", flush=True)
# the checks run on the CPU (exact resumption is not guaranteed by the non-deterministic GPU kernels)
sh(f"cd {CODE} && CUDA_VISIBLE_DEVICES= python -m script.check 2>&1 | grep -E '^(ok|all checks|Traceback|AssertionError)' ")

from huggingface_hub import HfApi  # noqa: E402

api = HfApi()
print("[hf] whoami:", api.whoami()["name"], flush=True)
info = api.repo_info(args.hf_repo)
print(f"[hf] repo {args.hf_repo} private={info.private}", flush=True)
assert info.private, "the repository is not private"

# 3) room files, image list, test gallery
for name, dest in DATA_FILES.items():
    fetch(args.hf_repo, args.data_branch, name, dest, expected)
print(f"[ok] {len(DATA_FILES)} files from {args.data_branch}, sha256 as listed", flush=True)

# 4) feature cache
os.makedirs(CACHE, exist_ok=True)
if not args.recompute_cache:
    for f in CACHE_FILES:
        fetch(args.hf_repo, CACHE_BRANCH, f, f"{CACHE}/{f}", expected)
    print(f"[ok] feature cache from {CACHE_BRANCH}, sha256 as listed", flush=True)
else:
    from huggingface_hub import snapshot_download

    want_f = {}
    with gzip.open(f"{KIT}/furnitures_sha256.txt.gz", "rt") as f:
        for line in f:
            h, name = line.split()
            want_f[name] = h
    os.makedirs(FURN, exist_ok=True)
    if len(glob.glob(f"{FURN}/*.jpg")) != len(want_f):
        snap = snapshot_download("byliu/DeepFurniture", repo_type="dataset", allow_patterns=["data/furnitures/*.tar.gz"])
        chunks = sorted(glob.glob(f"{snap}/data/furnitures/*.tar.gz"))
        print("chunks:", len(chunks), flush=True)
        tmp = "/content/_furn_extract"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for c in chunks:
            subprocess.run(["tar", "-xzf", c, "-C", tmp], check=True)
        n = 0
        for root, _, files in os.walk(tmp):
            for fn in files:
                if fn.endswith(".jpg"):
                    shutil.move(os.path.join(root, fn), os.path.join(FURN, fn))
                    n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        print("moved jpg:", n, flush=True)
    from hd_common import sha256  # noqa: E402

    names = sorted(os.path.basename(x) for x in glob.glob(f"{FURN}/*"))
    assert names == sorted(want_f), f"item images differ from the list: VM {len(names)} / list {len(want_f)}"
    bad = [n for n in names if sha256(f"{FURN}/{n}") != want_f[n]]
    assert not bad, f"sha256 of item images differs from the list: {len(bad)} (e.g. {bad[:5]})"
    print(f"[ok] {len(names)} item images, sha256 as listed", flush=True)
    sh(f"cd {CODE} && python -m script.data --image_dir {FURN} "
       f"--rooms {KIT}/rooms_train.json {KIT}/rooms_val.json {KIT}/rooms_test0.json --out {CACHE} 2>&1 | tail -8")
    if args.upload_cache:
        api.create_branch(args.hf_repo, branch=CACHE_BRANCH, exist_ok=True)
        for f in CACHE_FILES:
            api.upload_file(path_or_fileobj=f"{CACHE}/{f}", path_in_repo=f, repo_id=args.hf_repo, revision=CACHE_BRANCH)
        print(f"[ok] feature cache computed and put on {CACHE_BRANCH}", flush=True)
    else:
        print("[ok] feature cache computed (not uploaded)", flush=True)
meta = json.load(open(f"{CACHE}/items.json"))
sizes = {f: os.path.getsize(f"{CACHE}/{f}") for f in CACHE_FILES}
print(f"[cache] items={len(meta['ids'])} categories={meta['categories']} sizes={sizes}", flush=True)

# 5) public DiT-XL/2 weights
if not os.path.isfile(WEIGHTS):
    sh(f"wget -q {WEIGHTS_URL} -O {WEIGHTS}.tmp && mv {WEIGHTS}.tmp {WEIGHTS}")
check_file(WEIGHTS, f"url:{WEIGHTS_URL}", expected)
print(f"[weights] {os.path.getsize(WEIGHTS)} bytes, sha256 as listed", flush=True)

# 6) session record
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && df -h /content | tail -1 && free -g | sed -n 2p")
rec = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           gpu=torch.cuda.get_device_name(0), python=platform.python_version(), torch=torch.__version__,
           cuda=torch.version.cuda, code_rev=rev, cache_items=len(meta["ids"]))
os.makedirs(RUNS_DIR, exist_ok=True)
json.dump(rec, open(f"{RUNS_DIR}/colab_session.json", "w"), ensure_ascii=False, indent=1)
print(rec, flush=True)
print("[setup_vm] done", flush=True)
