"""
Rooms of furniture items for group training of a text-conditional DiT.

A room is one scene of the DeepFurniture retrieval benchmark, already split into query items and target items.
The targets of a room are denoised together as one group; the query items condition them, either as a pooled
image embedding or as clean members of the group.

Room file (JSON), written by the exporter of the retrieval code base:

    {"split": "train",
     "rooms": [{"scene_id": "...",
                "queries": [{"id": 123, "category": "table"}, ...],
                "targets": [{"id": 456, "category": "lamp", "sample_index": 0}, ...]}, ...]}

Run this file once to compute the features that training and generation read:

    python -m script.data --image_dir furnitures --rooms rooms_train.json rooms_val.json rooms_test.json --out cache
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger("GroupDiff")

PROMPT_TEMPLATE = "a {category}"
NULL_PROMPT = ""
LATENT_SCALE = 0.18215


def prompt_of(category: str) -> str:
    """Text prompt of an item. The category name is used as it is written in the room file."""
    return PROMPT_TEMPLATE.format(category=category)


@dataclass
class Room:
    scene_id: str
    query_ids: list[int]
    query_categories: list[str]
    target_ids: list[int]
    target_categories: list[str]
    sample_indices: list[int]

    @property
    def num_targets(self) -> int:
        return len(self.target_ids)

    @property
    def num_queries(self) -> int:
        return len(self.query_ids)


def load_rooms(path: str, limit: int = 0) -> list[Room]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rooms = []
    for r in data["rooms"]:
        rooms.append(
            Room(
                scene_id=str(r["scene_id"]),
                query_ids=[int(q["id"]) for q in r["queries"]],
                query_categories=[q["category"] for q in r["queries"]],
                target_ids=[int(t["id"]) for t in r["targets"]],
                target_categories=[t["category"] for t in r["targets"]],
                sample_indices=[int(t["sample_index"]) for t in r["targets"]],
            )
        )
    if limit > 0:
        rooms = rooms[:limit]
    return rooms


#################################################################################
#                              Planning of an epoch                             #
#################################################################################


@dataclass
class Batch:
    """
    One optimizer step.
    kind: "single" denoises every target alone, "group" denoises the targets of each room together.
    conditional: True or False for the whole batch, or None to drop the conditions at random.
    targets: (room index, target index) pairs of a "single" batch.
    rooms: room indices of a "group" batch.
    """

    kind: str
    conditional: bool | None
    targets: list[tuple[int, int]] = field(default_factory=list)
    rooms: list[int] = field(default_factory=list)

    def num_targets(self, rooms: list[Room]) -> int:
        if self.kind == "single":
            return len(self.targets)
        return sum(rooms[r].num_targets for r in self.rooms)


def _pack_rooms(order: list[int], rooms: list[Room], batch_size: int) -> list[list[int]]:
    """Fill batches with whole rooms in the given order, without exceeding batch_size targets."""
    batches, current, count = [], [], 0
    for r in order:
        n = rooms[r].num_targets
        if current and count + n > batch_size:
            batches.append(current)
            current, count = [], 0
        current.append(r)
        count += n
    if current:
        batches.append(current)
    return batches


def plan_epoch(
    rooms: list[Room], form: str, batch_size: int, seed: int, epoch: int, group_ratio: float = 0.1
) -> list[Batch]:
    """
    Split one pass over all targets into batches. batch_size counts targets (the images that carry a loss).

    none: every batch is "single"; conditions are dropped at random per target.
    f:    every batch is "group"; conditions are dropped at random per room (GroupDiff-f).
    l:    rooms holding group_ratio of the targets form unconditional "group" batches, all other targets form
          conditional "single" batches (GroupDiff-l).
    """
    rng = random.Random(f"{seed}-{epoch}")
    order = list(range(len(rooms)))
    rng.shuffle(order)

    if form == "f":
        return [Batch("group", None, rooms=b) for b in _pack_rooms(order, rooms, batch_size)]

    group_rooms = []
    if form == "l":
        total = sum(r.num_targets for r in rooms)
        count = 0
        while order and count < group_ratio * total:
            r = order.pop()
            group_rooms.append(r)
            count += rooms[r].num_targets
    elif form != "none":
        raise ValueError(f"Unsupported form {form}")

    pairs = [(r, j) for r in order for j in range(rooms[r].num_targets)]
    rng.shuffle(pairs)
    single_conditional = True if form == "l" else None
    batches = [
        Batch("single", single_conditional, targets=pairs[i : i + batch_size]) for i in range(0, len(pairs), batch_size)
    ]
    batches += [Batch("group", False, rooms=b) for b in _pack_rooms(group_rooms, rooms, batch_size)]
    rng.shuffle(batches)
    return batches


#################################################################################
#                                Feature cache                                  #
#################################################################################


@dataclass
class GroupInputs:
    """
    Inputs of R groups that have the same number of targets and of query members.
    Members are ordered targets first, then query members.
    """

    num_targets: int
    num_queries: int
    target_latents: torch.Tensor  # (R * num_targets, C, H, W), clean
    query_latents: torch.Tensor | None  # (R, num_queries, C, H, W), clean
    text: torch.Tensor  # (R * (num_targets + num_queries), L, D) text token features
    image_embeds: torch.Tensor | None  # (R * (num_targets + num_queries), E) pooled image embedding of the query set
    sample_indices: list[int]  # one per target, in the order of target_latents
    target_ids: list[int]
    scene_ids: list[str]

    @property
    def num_groups(self) -> int:
        return self.target_latents.shape[0] // self.num_targets


class FeatureCache:
    """Latents, pooled image embeddings and text token features computed by `python -m script.data`."""

    def __init__(self, cache_dir: str, device: torch.device | str = "cpu"):
        with open(os.path.join(cache_dir, "items.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.row = {int(i): r for r, i in enumerate(self.meta["ids"])}
        self.device = torch.device(device)
        self.latents = torch.from_numpy(np.load(os.path.join(cache_dir, "latents.npy"))).to(self.device)
        flip_path = os.path.join(cache_dir, "latents_flip.npy")
        self.latents_flip = torch.from_numpy(np.load(flip_path)).to(self.device) if os.path.isfile(flip_path) else None
        self.image_embeds = torch.from_numpy(np.load(os.path.join(cache_dir, "image_embeds.npy"))).to(self.device)
        text = torch.load(os.path.join(cache_dir, "text_embeds.pt"), map_location="cpu")
        self.prompt_row = {p: i for i, p in enumerate(text["prompts"])}
        self.text_embeds = text["embeds"].to(self.device)
        self._query_embeddings: dict[str, torch.Tensor] = {}

    def rows(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor([self.row[i] for i in ids], dtype=torch.long, device=self.device)

    def item_latents(self, ids: list[int], flip: torch.Tensor | None = None) -> torch.Tensor:
        rows = self.rows(ids)
        latents = self.latents[rows].float()
        if flip is not None:
            assert self.latents_flip is not None, "the cache has no flipped latents; compute it with --flip"
            latents = torch.where(flip.view(-1, 1, 1, 1), self.latents_flip[rows].float(), latents)
        return latents

    def query_embedding(self, room: Room) -> torch.Tensor:
        """Mean of the pooled image embeddings of the query items of a room."""
        if room.scene_id not in self._query_embeddings:
            self._query_embeddings[room.scene_id] = self.image_embeds[self.rows(room.query_ids)].float().mean(dim=0)
        return self._query_embeddings[room.scene_id]

    def text_of(self, prompts: list[str]) -> torch.Tensor:
        rows = torch.tensor([self.prompt_row[p] for p in prompts], dtype=torch.long, device=self.device)
        return self.text_embeds[rows].float()

    def build(
        self,
        rooms: list[Room],
        entries: list[tuple[int, list[int], bool, bool]],
        use_image_embeds: bool,
        random_flip: bool = False,
    ) -> GroupInputs:
        """
        Build the inputs of groups that share one shape.
        entries: (room index, target indices of the group, with query members, conditional) per group.
        """
        num_targets = len(entries[0][1])
        num_queries = rooms[entries[0][0]].num_queries if entries[0][2] else 0
        target_ids, query_ids, prompts, sample_indices, scene_ids, embeds = [], [], [], [], [], []
        for r, target_idx, with_queries, conditional in entries:
            room = rooms[r]
            assert len(target_idx) == num_targets and (room.num_queries if with_queries else 0) == num_queries
            ids = [room.target_ids[j] for j in target_idx]
            target_ids += ids
            sample_indices += [room.sample_indices[j] for j in target_idx]
            scene_ids += [room.scene_id] * num_targets
            categories = [room.target_categories[j] for j in target_idx]
            if with_queries:
                query_ids += room.query_ids
                categories += room.query_categories
            prompts += [prompt_of(c) if conditional else NULL_PROMPT for c in categories]
            if use_image_embeds:
                embed = self.query_embedding(room) if conditional else torch.zeros_like(self.image_embeds[0]).float()
                embeds.append(embed.expand(len(categories), -1))

        def flips(n):
            return (torch.rand(n, device=self.device) < 0.5) if random_flip else None

        target_latents = self.item_latents(target_ids, flips(len(target_ids)))
        query_latents = None
        if num_queries:
            query_latents = self.item_latents(query_ids, flips(len(query_ids)))
            query_latents = query_latents.view(len(entries), num_queries, *query_latents.shape[1:])
        return GroupInputs(
            num_targets=num_targets,
            num_queries=num_queries,
            target_latents=target_latents,
            query_latents=query_latents,
            text=self.text_of(prompts),
            image_embeds=torch.cat(embeds, dim=0) if use_image_embeds else None,
            sample_indices=sample_indices,
            target_ids=target_ids,
            scene_ids=scene_ids,
        )


def group_entries(
    batch: Batch,
    rooms: list[Room],
    query_members: bool,
    query_member_drop: float,
    cond_drop: float,
    generator: torch.Generator,
) -> list[list[tuple[int, list[int], bool, bool]]]:
    """
    Turn a batch into lists of groups of one shape each. The random choices (dropping the conditions, dropping the
    query members) are drawn from the generator, so a step can be repeated exactly.
    """

    def draw() -> float:
        return float(torch.rand((), generator=generator))

    if batch.kind == "single":
        entries = []
        for r, j in batch.targets:
            conditional = batch.conditional if batch.conditional is not None else draw() >= cond_drop
            entries.append((r, [j], False, conditional))
        return [entries]

    by_shape: dict[tuple[int, int], list] = {}
    for r in batch.rooms:
        room = rooms[r]
        conditional = batch.conditional if batch.conditional is not None else draw() >= cond_drop
        with_queries = query_members and draw() >= query_member_drop
        shape = (room.num_targets, room.num_queries if with_queries else 0)
        by_shape.setdefault(shape, []).append((r, list(range(room.num_targets)), with_queries, conditional))
    return [by_shape[k] for k in sorted(by_shape)]


#################################################################################
#                             Computing the features                            #
#################################################################################


def collect_items(room_files: list[str]) -> tuple[list[int], list[str]]:
    ids, categories = set(), set()
    for path in room_files:
        for room in load_rooms(path):
            ids.update(room.query_ids)
            ids.update(room.target_ids)
            categories.update(room.query_categories)
            categories.update(room.target_categories)
    return sorted(ids), sorted(categories)


@torch.no_grad()
def compute_features(args):
    from diffusers.models import AutoencoderKL
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

    from utils.loader import center_crop_arr

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ids, categories = collect_items(args.rooms)
    logger.info("items: %d, categories: %s", len(ids), categories)

    def load(i):
        return Image.open(os.path.join(args.image_dir, f"{i}.jpg")).convert("RGB")

    # latents: one posterior sample per image, scaled as in DiT
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents, latents_flip = [], []
    for start in range(0, len(ids), args.batch_size):
        images = [center_crop_arr(load(i), args.image_size) for i in ids[start : start + args.batch_size]]
        x = torch.from_numpy(np.stack([np.asarray(im) for im in images])).to(device)
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        latents.append((vae.encode(x).latent_dist.sample(generator=generator) * LATENT_SCALE).cpu().numpy())
        if args.flip:
            z = vae.encode(x.flip(dims=[3])).latent_dist.sample(generator=generator) * LATENT_SCALE
            latents_flip.append(z.cpu().numpy())
        if start % (args.batch_size * 20) == 0:
            logger.info("latents %d / %d", start, len(ids))
    np.save(os.path.join(args.out, "latents.npy"), np.concatenate(latents).astype(np.float32))
    if args.flip:
        np.save(os.path.join(args.out, "latents_flip.npy"), np.concatenate(latents_flip).astype(np.float32))
    del vae

    # pooled image embeddings, computed as IP-Adapter computes them
    processor = CLIPImageProcessor()
    encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder, subfolder=args.image_encoder_subfolder)
    encoder = encoder.to(device).eval()
    embeds = []
    for start in range(0, len(ids), args.batch_size):
        images = [load(i) for i in ids[start : start + args.batch_size]]
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
        embeds.append(encoder(pixels).image_embeds.float().cpu().numpy())
        if start % (args.batch_size * 20) == 0:
            logger.info("image embeddings %d / %d", start, len(ids))
    np.save(os.path.join(args.out, "image_embeds.npy"), np.concatenate(embeds).astype(np.float32))
    del encoder

    # text token features of the null prompt and of one prompt per category
    prompts = [NULL_PROMPT] + [prompt_of(c) for c in categories]
    tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder).to(device).eval()
    tokens = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    text_embeds = text_encoder(tokens.input_ids.to(device)).last_hidden_state.float().cpu()
    torch.save({"prompts": prompts, "embeds": text_embeds}, os.path.join(args.out, "text_embeds.pt"))

    meta = {
        "ids": ids,
        "categories": categories,
        "image_size": args.image_size,
        "vae": args.vae,
        "image_encoder": f"{args.image_encoder}/{args.image_encoder_subfolder}",
        "text_encoder": args.text_encoder,
        "flip": bool(args.flip),
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "items.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info("wrote %s: latents %s, text %s", args.out, np.concatenate(latents).shape, tuple(text_embeds.shape))


def get_args_parser():
    parser = argparse.ArgumentParser("Compute the features of the items of the room files")
    parser.add_argument("--image_dir", required=True, type=str, help="directory of <item id>.jpg")
    parser.add_argument("--rooms", required=True, nargs="+", type=str, help="room files")
    parser.add_argument("--out", required=True, type=str, help="cache directory to write")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--flip", action="store_true", help="also store the latents of the mirrored images")
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse", type=str)
    parser.add_argument("--image_encoder", default="h94/IP-Adapter", type=str)
    parser.add_argument("--image_encoder_subfolder", default="sdxl_models/image_encoder", type=str)
    parser.add_argument("--text_encoder", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    compute_features(get_args_parser().parse_args())
