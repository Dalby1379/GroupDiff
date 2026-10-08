"""
Recall@K and category accuracy of the generated targets (clip only).

Ported from the retrieval code base (SetRetrieval, methods/heterogeneous_diffusion/evaluate.py at 2aa1dda), which
scored the runs of this directory. The computation is unchanged; the style measure (DreamSim) and the comparison
sub-command are left out.

    - The generated images are embedded with the IP-Adapter image encoder (OpenCLIP ViT-bigG/14, image_embeds) and
      L2 normalized.
    - They are ranked by inner product against the test gallery of the target category (one pkl per category,
      key "clip_vectors"), in descending order.
    - A sample is a hit at K when the rank of its target is within max(1, ceil(gallery size * K / 100)), K = 1, 5, 10, 20.
    - Category accuracy is the share of samples whose nearest gallery item over all categories has the target category.

Sub-commands:
    rank   ranks, Recall@K and category accuracy from samples.jsonl and clip_embeds.npy (CPU)
    embed  clip embeddings of the generated images (GPU)

Example::

    python -m script.evaluate embed --result work_dirs/eval/test0_members0
    python -m script.evaluate rank --result work_dirs/eval/test0_members0 --clip_gallery kit/clip_vectors
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle

import numpy as np

TOPK = [1, 5, 10, 20]
IMAGE_ENCODER = ("h94/IP-Adapter", "sdxl_models/image_encoder")

Gallery = dict[str, tuple[np.ndarray, np.ndarray]]  # category name (lower case) -> (identity ids, normalized features)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def topk_percent_threshold(n_items: int, k_percent: int) -> int:
    return max(1, int(math.ceil(n_items * (k_percent / 100.0))))


def load_gallery(root: str, key: str) -> Gallery:
    """Reads one pkl per category. Token features (3-D) are averaged before normalization."""
    gallery: Gallery = {}
    for path in sorted(glob.glob(os.path.join(root, "*.pkl"))):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        name = obj.get("category_name") or obj.get("category") or os.path.splitext(os.path.basename(path))[0]
        vectors = np.asarray(obj[key]).astype(np.float32)
        if vectors.ndim == 3:
            vectors = vectors.mean(axis=1)
        gallery[str(name).lower()] = (np.asarray(obj["identity_ids"]).astype(np.int64), l2norm(vectors))
    return gallery


def gallery_from_items(ids: list[int], embeds: np.ndarray, category_of: dict[int, str]) -> Gallery:
    """A gallery from item embeddings (for items that are not in the test gallery, such as the validation rooms)."""
    by_category: dict[str, list[int]] = {}
    for row, i in enumerate(ids):
        if int(i) in category_of:
            by_category.setdefault(category_of[int(i)], []).append(row)
    return {c: (np.asarray([ids[r] for r in rows], dtype=np.int64), l2norm(embeds[rows].astype(np.float32)))
            for c, rows in by_category.items()}


def rank_of(query: np.ndarray, gallery: Gallery, category: str, target_id: int) -> tuple[int, int]:
    """(rank of the target, gallery size), in descending order of the inner product."""
    ids, vectors = gallery[category]
    matches = np.where(ids == int(target_id))[0]
    assert len(matches) == 1, f"target missing from the gallery or duplicated: {category} {target_id}"
    order = np.argsort(-(vectors @ query))
    return int(np.where(order == matches[0])[0][0]) + 1, len(ids)


def nearest_category(query: np.ndarray, gallery: Gallery) -> str:
    """Category of the nearest item over the galleries of all categories."""
    best, best_sim = "", -np.inf
    for category in sorted(gallery):
        sim = float((gallery[category][1] @ query).max())
        if sim > best_sim:
            best, best_sim = category, sim
    return best


def read_samples(result_dir: str) -> list[dict]:
    with open(os.path.join(result_dir, "samples.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rank_results(result_dir: str, clip_gallery: Gallery, out_dir: str = "") -> dict:
    samples = read_samples(result_dir)
    clip = l2norm(np.load(os.path.join(result_dir, "clip_embeds.npy")).astype(np.float32))
    assert len(samples) == len(clip), "samples.jsonl and clip_embeds.npy differ in length"
    assert len({s["sample_index"] for s in samples}) == len(samples), "duplicated sample_index"

    rows = []
    for k, s in enumerate(samples):
        category = s["category"].lower()
        rank_clip, n_cat = rank_of(clip[k], clip_gallery, category, s["target_id"])
        predicted = nearest_category(clip[k], clip_gallery)
        rows.append({"sample_index": s["sample_index"], "scene_id": s["scene_id"], "positive_cat": category, "positive_id": s["target_id"],
                     "n_cat": n_cat, "rank_clip": rank_clip, "nearest_category": predicted, "category_correct": int(predicted == category)})
    rows.sort(key=lambda r: r["sample_index"])

    out_dir = out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_sample_results.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {"count": len(rows)}
    with open(os.path.join(out_dir, "summary_topk.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "k", "correct", "count", "accuracy"])
        for k in TOPK:
            correct = sum(r["rank_clip"] <= topk_percent_threshold(r["n_cat"], k) for r in rows)
            writer.writerow(["clip", k, correct, len(rows), 100.0 * correct / len(rows)])
            summary[f"recall_clip@{k}"] = 100.0 * correct / len(rows)
    categories = sorted({r["positive_cat"] for r in rows})
    with open(os.path.join(out_dir, "summary_category.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "correct", "count", "accuracy", "recall_clip@1"])
        for category in ["all"] + categories:
            part = [r for r in rows if category == "all" or r["positive_cat"] == category]
            correct = sum(r["category_correct"] for r in part)
            hit = sum(r["rank_clip"] <= topk_percent_threshold(r["n_cat"], 1) for r in part)
            writer.writerow([category, correct, len(part), 100.0 * correct / len(part), 100.0 * hit / len(part)])
    summary["category_correct"] = 100.0 * sum(r["category_correct"] for r in rows) / len(rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def embed_images(result_dir: str, device: str = "cuda", batch_size: int = 32) -> None:
    """Clip embeddings of the images in samples.jsonl, saved next to it as clip_embeds.npy."""
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    samples = read_samples(result_dir)
    images = [Image.open(os.path.join(result_dir, s["image"])).convert("RGB") for s in samples]
    processor = CLIPImageProcessor()
    encoder = CLIPVisionModelWithProjection.from_pretrained(IMAGE_ENCODER[0], subfolder=IMAGE_ENCODER[1]).to(device, dtype=torch.float16).eval()
    clip = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            pixels = processor(images=images[start : start + batch_size], return_tensors="pt").pixel_values.to(device, dtype=torch.float16)
            clip.append(encoder(pixels).image_embeds.float().cpu().numpy())
    np.save(os.path.join(result_dir, "clip_embeds.npy"), l2norm(np.concatenate(clip)))
    print(f"embedded {len(images)} images (clip) -> {result_dir}")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recall@K and category accuracy (clip)")
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("rank", help="ranks, Recall@K and category accuracy")
    rank.add_argument("--result", required=True, type=str, help="directory with samples.jsonl and clip_embeds.npy")
    rank.add_argument("--out", default="", type=str, help="defaults to --result")
    rank.add_argument("--clip_gallery", default="", type=str, help="directory of the per-category test gallery pkl files")
    rank.add_argument("--item_gallery", default="", type=str,
                      help="instead of the test gallery, build the gallery from this feature cache (items.json, image_embeds.npy)")
    rank.add_argument("--item_gallery_rooms", default=[], nargs="*", type=str, help="room files that choose the items of --item_gallery")
    embed = sub.add_parser("embed", help="clip embeddings of the generated images")
    embed.add_argument("--result", required=True, type=str)
    embed.add_argument("--device", default="cuda", type=str)
    return parser


def main(args) -> None:
    if args.command == "embed":
        embed_images(args.result, args.device)
        return
    if args.item_gallery:
        from .data import load_rooms

        category_of: dict[int, str] = {}
        for path in args.item_gallery_rooms:
            for room in load_rooms(path):
                category_of.update(zip(room.query_ids + room.target_ids, room.query_categories + room.target_categories))
        with open(os.path.join(args.item_gallery, "items.json"), encoding="utf-8") as f:
            ids = json.load(f)["ids"]
        clip_gallery = gallery_from_items(ids, np.load(os.path.join(args.item_gallery, "image_embeds.npy")), category_of)
    else:
        assert args.clip_gallery, "give --clip_gallery (test gallery) or --item_gallery"
        clip_gallery = load_gallery(args.clip_gallery, "clip_vectors")
    summary = rank_results(args.result, clip_gallery, args.out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(get_args_parser().parse_args())
