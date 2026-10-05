"""
ImageNet-1k classes of furniture items.

The released GroupDiff weights take an ImageNet-1k class label as their only condition. To generate furniture
with them, a furniture category has to be expressed as ImageNet-1k classes. This script runs ImageNet-1k
classifiers on the item images of the DeepFurniture retrieval benchmark and reports, for every furniture
category, the classes its items are assigned to.

    # 1. top classes of every item image, one file per classifier
    python -m script.imagenet_labels classify --image_dir furnitures --metadata furnitures.jsonl \
        --categories categories.json --out labels

    # 2. classes per furniture category, over the items of the given rooms (all items without --rooms)
    python -m script.imagenet_labels report --out labels --rooms rooms_train.json --tag train

    # 3. group file for script/generate_imagenet.py: the targets of every room, each with the class of its category
    python -m script.imagenet_labels groups --classes labels/classes_train_<classifier>.json \
        --rooms rooms_test.json --out groups_rooms.json

The report step also writes classes_<tag>_<classifier>.json, the class with the largest share for every category.
It is a plain {category: class} file and can be edited before the groups step. Without --rooms, the groups step
writes groups of one class instead (--groups_per_category groups of --group_size images for every category).

The class indices are those of the classifiers, which follow the sorted WordNet ids of ImageNet-1k. The label
embedding of the released weights uses the same indices (dataset/extract_feats.py reads the classes with
torchvision's ImageFolder).
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("GroupDiff")

TOP_K = 10


def load_items(metadata: str, categories: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Item ids, the category index of every item, and the category names in the order of their ids."""
    with open(categories, encoding="utf-8") as f:
        names = json.load(f)
    category_ids = sorted(names, key=int)
    index_of = {int(c): i for i, c in enumerate(category_ids)}
    ids, cats = [], []
    with open(metadata, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ids.append(str(item["furniture_id"]))
            cats.append(index_of[int(item["category_id"])])
    return ids, np.asarray(cats, dtype=np.int16), [names[c] for c in category_ids]


class ItemImages(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, ids: list[str], transform):
        self.image_dir, self.ids, self.transform = image_dir, ids, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(os.path.join(self.image_dir, f"{self.ids[i]}.jpg")) as image:
            return self.transform(image.convert("RGB"))


@torch.no_grad()
def classify(args):
    import timm

    ids, cats, names = load_items(args.metadata, args.categories)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in args.models:
        path = os.path.join(args.out, f"{name}.npz")
        if os.path.isfile(path):
            logger.info("%s: found %s, skipped", name, path)
            continue
        model = timm.create_model(name, pretrained=True).to(device).eval()
        config = timm.data.resolve_data_config(model=model)
        # the whole item image is resized, without a centre crop that would cut the edges of an item
        config["crop_pct"] = 1.0
        loader = torch.utils.data.DataLoader(
            ItemImages(args.image_dir, ids, timm.data.create_transform(**config)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        top_index, top_prob = [], []
        for n, images in enumerate(loader):
            prob = model(images.to(device)).float().softmax(dim=-1)
            assert prob.shape[1] == 1000, f"{name} is not an ImageNet-1k classifier"
            p, k = prob.topk(TOP_K, dim=-1)
            top_index.append(k.cpu().numpy().astype(np.int16))
            top_prob.append(p.cpu().numpy().astype(np.float16))
            if n % 20 == 0:
                logger.info("%s: %d / %d batches", name, n, len(loader))
        np.savez(
            path,
            ids=np.asarray(ids),
            category=cats,
            category_names=np.asarray(names),
            top_index=np.concatenate(top_index),
            top_prob=np.concatenate(top_prob),
        )
        logger.info("%s: wrote %s", name, path)
        del model


def class_names() -> list[str]:
    from timm.data import ImageNetInfo

    info = ImageNetInfo("imagenet-1k")
    return [info.index_to_description(i) for i in range(1000)]


def room_item_ids(paths: list[str]) -> set[str]:
    ids = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for room in json.load(f)["rooms"]:
                ids.update(str(item["id"]) for item in room["queries"] + room["targets"])
    return ids


def category_table(data, keep: np.ndarray) -> dict:
    """Per furniture category: how its items are spread over the classes.

    top1 and top5 count the items whose first, or first five, classes contain the class; prob is the mean
    probability of the class (classes below the stored first ten count as zero). purity is, of the items of all
    categories whose first class is the class, the share that belongs to this category.
    """
    table = {}
    first_class_items = np.bincount(data["top_index"][keep, 0].astype(np.int64), minlength=1000)
    for c, name in enumerate(data["category_names"]):
        rows = keep & (data["category"] == c)
        index, prob = data["top_index"][rows].astype(np.int64), data["top_prob"][rows].astype(np.float64)
        n = int(rows.sum())
        top1 = np.bincount(index[:, 0], minlength=1000) / max(n, 1)
        top5 = np.bincount(index[:, :5].ravel(), minlength=1000) / max(n, 1)
        mean_prob = np.bincount(index.ravel(), weights=prob.ravel(), minlength=1000) / max(n, 1)
        table[str(name)] = {
            "items": n,
            "top1": top1,
            "top5": top5,
            "prob": mean_prob,
            "purity": np.bincount(index[:, 0], minlength=1000) / np.maximum(first_class_items, 1),
            "confidence": float(prob[:, 0].mean()) if n else 0.0,
        }
    return table


def report(args):
    names = class_names()
    files = sorted(f for f in os.listdir(args.out) if f.endswith(".npz"))
    models = [f[: -len(".npz")] for f in files]
    data = [np.load(os.path.join(args.out, f)) for f in files]
    ids = data[0]["ids"]
    if args.rooms:
        wanted = room_item_ids(args.rooms)
        keep = np.asarray([i in wanted for i in ids])
    else:
        keep = np.ones(len(ids), dtype=bool)
    tables = [category_table(d, keep) for d in data]

    lines = [
        f"# ImageNet-1k classes of furniture items ({args.tag})",
        "",
        f"Items: {int(keep.sum())} of {len(ids)}"
        + (f" (those of {', '.join(os.path.basename(p) for p in args.rooms)})" if args.rooms else ""),
        "",
        "share = items of the category whose first class is the class; in top 5 = items with the class among their",
        "first five; prob = mean probability of the class; purity = of the items of all categories whose first class",
        f"is the class, the share that belongs to this category. Classes are listed by share, the first "
        f"{args.classes_per_category} of every category.",
    ]
    summary = {"tag": args.tag, "items": int(keep.sum()), "models": models, "categories": {}}
    for category in tables[0]:
        lines += ["", f"## {category} ({tables[0][category]['items']} items)", ""]
        summary["categories"][category] = {"items": tables[0][category]["items"]}
        for model, table in zip(models, tables):
            entry = table[category]
            order = np.argsort(-entry["top1"])[: args.classes_per_category]
            lines += [
                f"{model}: mean probability of the first class {entry['confidence']:.2f}",
                "",
                "| class | name | share | in top 5 | prob | purity |",
                "|---|---|---|---|---|---|",
            ]
            for k in order:
                lines.append(
                    f"| {k} | {names[k]} | {100 * entry['top1'][k]:.1f}% | {100 * entry['top5'][k]:.1f}% "
                    f"| {entry['prob'][k]:.3f} | {100 * entry['purity'][k]:.0f}% |"
                )
            lines.append("")
            summary["categories"][category][model] = {
                "confidence": entry["confidence"],
                "classes": [
                    {
                        "index": int(k),
                        "name": names[k],
                        "top1": float(entry["top1"][k]),
                        "top5": float(entry["top5"][k]),
                        "prob": float(entry["prob"][k]),
                        "purity": float(entry["purity"][k]),
                    }
                    for k in order
                ],
            }
    if len(data) > 1:
        same = (data[0]["top_index"][keep, 0] == data[1]["top_index"][keep, 0]).mean()
        lines += ["", f"The first classes of {models[0]} and {models[1]} agree on {100 * same:.1f}% of the items."]
        summary["first_class_agreement"] = float(same)

    for model, table in zip(models, tables):
        first = {category: int(np.argmax(entry["top1"])) for category, entry in table.items()}
        with open(os.path.join(args.out, f"classes_{args.tag}_{model}.json"), "w", encoding="utf-8") as f:
            json.dump(first, f, indent=1)
    with open(os.path.join(args.out, f"report_{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, f"report_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    logger.info("wrote report_%s.md and report_%s.json in %s", args.tag, args.tag, args.out)


def groups(args):
    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)
    entries, skipped = [], 0
    if args.rooms:
        with open(args.rooms, encoding="utf-8") as f:
            rooms = json.load(f)["rooms"]
        for room in rooms:
            targets = room["targets"]
            # the sample embedding of the weights bounds the size of a group; larger rooms are left out, not cut
            if len(targets) > args.max_group_size:
                skipped += 1
                continue
            entries.append({
                "name": room["scene_id"],
                "labels": [classes[t["category"]] for t in targets],
                "members": [f"{t['id']}-{t['category'].replace('#', '+')}" for t in targets],
            })
        logger.info("%d rooms, %d left out with more than %d targets", len(entries), skipped, args.max_group_size)
    else:
        for category, label in classes.items():
            for n in range(args.groups_per_category):
                entries.append({
                    "name": f"{category.replace('#', '+')}-{n}",
                    "labels": [label] * args.group_size,
                    "members": [str(k) for k in range(args.group_size)],
                })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "groups": entries}, f, indent=1)
    logger.info("wrote %d groups, %d images, to %s", len(entries), sum(len(e["labels"]) for e in entries), args.out)


def get_args_parser():
    parser = argparse.ArgumentParser("ImageNet-1k classes of furniture items")
    steps = parser.add_subparsers(dest="step", required=True)

    step = steps.add_parser("classify")
    step.add_argument("--image_dir", required=True, type=str, help="directory of <furniture_id>.jpg")
    step.add_argument("--metadata", required=True, type=str, help="furnitures.jsonl")
    step.add_argument("--categories", required=True, type=str, help="categories.json")
    step.add_argument("--out", required=True, type=str)
    step.add_argument("--models", nargs="+", type=str, help="timm names of ImageNet-1k classifiers",
                      default=["resnet50.tv2_in1k", "convnext_base.fb_in22k_ft_in1k"])
    step.add_argument("--batch_size", default=256, type=int)
    step.add_argument("--num_workers", default=8, type=int)

    step = steps.add_parser("report")
    step.add_argument("--out", required=True, type=str, help="directory written by the classify step")
    step.add_argument("--rooms", nargs="*", type=str, default=[], help="room files that select the items")
    step.add_argument("--tag", default="all", type=str, help="name of the report")
    step.add_argument("--classes_per_category", default=8, type=int)

    step = steps.add_parser("groups")
    step.add_argument("--classes", required=True, type=str, help="{category: class}, e.g. from the report step")
    step.add_argument("--out", required=True, type=str, help="group file to write")
    step.add_argument("--rooms", default="", type=str, help="room file; one group for the targets of every room")
    step.add_argument("--max_group_size", default=4, type=int, help="rooms with more targets are left out")
    step.add_argument("--groups_per_category", default=1, type=int, help="without --rooms: groups of one class")
    step.add_argument("--group_size", default=4, type=int, help="without --rooms: images of a group")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = get_args_parser().parse_args()
    {"classify": classify, "report": report, "groups": groups}[arguments.step](arguments)
