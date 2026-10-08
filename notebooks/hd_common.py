"""Constants and helpers shared by the notebook scripts. They run on the Colab VM, from /content/GroupDiff.

The paths are the ones the recorded runs used, so that their records (generate_args.json, args.json) compare directly.
"""
import hashlib
import json
import os
import subprocess
import time

# run name -> (form, query_members, query_embedding)
RUNS = {
    "dit_embed_none": ("none", 0, 1), "dit_embed_l": ("l", 0, 1), "dit_embed_f": ("f", 0, 1),
    "dit_members_l": ("l", 1, 0), "dit_members_f": ("f", 1, 0),
}

CODE = "/content/GroupDiff"
KIT = "/content/kit"                # room files, the image list and the test gallery
KIT_EVAL = "/content/kit_eval"      # the 100 validation rooms for choosing the guidance weight
CACHE = "/content/cache"            # feature cache (latents, image and text embeddings)
FURN = "/content/furnitures"        # item images, only when the cache is computed again
WEIGHTS = "/content/DiT-XL-2-256x256.pt"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
RUNS_DIR = "/content/runs"
EVAL_DIR = "/content/eval"

HF_REPO = "Dalby123/heterogeneous-diffusion"   # private
DATA_BRANCH = "data-deepfurniture"
CACHE_BRANCH = "cache-dit"
CACHE_FILES = ["items.json", "latents.npy", "image_embeds.npy", "text_embeds.pt"]
# file on the data branch -> path on the VM
DATA_FILES = {
    "rooms_train.json": f"{KIT}/rooms_train.json",
    "rooms_val.json": f"{KIT}/rooms_val.json",
    "rooms_test0.json": f"{KIT}/rooms_test0.json",
    "furnitures_sha256.txt.gz": f"{KIT}/furnitures_sha256.txt.gz",
    "rooms_val100.json": f"{KIT_EVAL}/rooms_val100.json",
}
GALLERY_CATEGORIES = ["Bed", "Cabinet#Shelf", "Chair#Stool", "Curtain", "Decoration", "Door", "Home-appliance", "Lamp", "Plant",
                      "Sofa", "Table"]
DATA_FILES.update({f"clip_vectors/{c}.pkl": f"{KIT}/clip_vectors/{c}.pkl" for c in GALLERY_CATEGORIES})

SHA_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_files.sha256")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_sha256() -> dict:
    """'<revision>:<path>' (or 'url:<url>') -> sha256, from hf_files.sha256."""
    out = {}
    with open(SHA_LIST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                digest, key = line.split(None, 1)
                out[key] = digest
    return out


def check_file(path: str, key: str, expected: dict) -> None:
    assert key in expected, f"no sha256 for {key} in {SHA_LIST}"
    got = sha256(path)
    assert got == expected[key], f"sha256 mismatch: {key} ({got[:16]} != {expected[key][:16]})"


def fetch(repo: str, revision: str, name: str, dest: str, expected: dict) -> None:
    """Downloads one file of the private repository to dest (skipped when dest already matches) and checks its sha256."""
    key = f"{revision}:{name}"
    if os.path.isfile(dest) and sha256(dest) == expected.get(key):
        return
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    path = hf_hub_download(repo, name, revision=revision)
    tmp = dest + ".part"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        for b in iter(lambda: src.read(1 << 20), b""):
            dst.write(b)
    os.replace(tmp, dest)
    check_file(dest, key, expected)


def sh(cmd: str, tail: int = 15) -> None:
    """Runs a shell command, streams its output and keeps the last lines."""
    print(f"$ {cmd[:200]}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        lines.append(line.rstrip())
        if len(lines) % 50 == 0:
            print("  ...", lines[-1][:200], flush=True)
    rc = p.wait()
    print("\n".join(lines[-tail:]), flush=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)
    if rc:
        raise SystemExit(f"[ERR] exit {rc}: {cmd[:200]}")


def add_categories(out_dir: str, rooms_path: str) -> int:
    """script/generate.py does not write the category into samples.jsonl; the ranks need it, so it is added from the
    room file. Returns the number of samples (checked against the room file)."""
    with open(rooms_path, encoding="utf-8") as f:
        category = {t["sample_index"]: t["category"] for room in json.load(f)["rooms"] for t in room["targets"]}
    with open(f"{out_dir}/samples.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == len(category) and {r["sample_index"] for r in rows} == set(category), f"generated count differs from the room file: {out_dir}"
    if any("category" not in r for r in rows):
        with open(f"{out_dir}/samples.jsonl.tmp", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({**r, "category": category[r["sample_index"]]}) + "\n" for r in rows)
        os.replace(f"{out_dir}/samples.jsonl.tmp", f"{out_dir}/samples.jsonl")
    return len(rows)


def choose_cfg(cfgs: list, table: dict) -> float:
    """The candidate with the best clip Recall@1% on the validation rooms; ties go to the earlier (smaller) candidate."""
    return max(cfgs, key=lambda c: (table[str(c)]["recall_clip@1"], -cfgs.index(c)))
