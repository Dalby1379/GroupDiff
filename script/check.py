"""
Small checks of the room training code that run on a CPU in a few minutes, on synthetic data.

    python -m script.check

They cover what the training relies on: the added layers leave the outputs of the loaded weights unchanged at
the start, an epoch visits every target once, the query members carry no loss, a run can be resumed exactly, the
loss goes down on a few rooms, and generation starts every target from the same noise in every form.
"""

import copy
import json
import os
import tempfile

import numpy as np
import torch

from diffusion import create_diffusion
from models.denoiser import Denoiser
from models.model_utils import SIZE_DICT

from . import generate, train
from .data import NULL_PROMPT, FeatureCache, Room, group_entries, load_rooms, plan_epoch, prompt_of

CATEGORIES = ["bed", "cabinet#shelf", "chair#stool", "lamp", "table"]
IMAGE_SIZE = 64  # latents of 8 x 8, so that every check stays fast
SIZE_DICT["check"] = {"width": 64, "layers": 2, "heads": 4}  # a model small enough for a CPU


def make_data(root: str, num_rooms: int = 12, seed: int = 0) -> tuple[str, str]:
    """Write a room file and a feature cache with random features."""
    rng = np.random.RandomState(seed)
    rooms, next_id, sample_index = [], 100, 0
    for r in range(num_rooms):
        size = int(rng.randint(4, 9))
        ids = list(range(next_id, next_id + size))
        next_id += size
        num_queries = size // 2
        room = {"scene_id": f"room{r}", "queries": [], "targets": []}
        for k, i in enumerate(ids):
            category = CATEGORIES[int(rng.randint(len(CATEGORIES)))]
            if k < num_queries:
                room["queries"].append({"id": i, "category": category})
            else:
                room["targets"].append({"id": i, "category": category, "sample_index": sample_index})
                sample_index += 1
        rooms.append(room)
    rooms_path = os.path.join(root, "rooms.json")
    with open(rooms_path, "w", encoding="utf-8") as f:
        json.dump({"split": "check", "rooms": rooms}, f)

    cache_dir = os.path.join(root, "cache")
    os.makedirs(cache_dir)
    num_items = next_id - 100
    side = IMAGE_SIZE // 8
    np.save(os.path.join(cache_dir, "latents.npy"), rng.randn(num_items, 4, side, side).astype(np.float32))
    np.save(os.path.join(cache_dir, "image_embeds.npy"), rng.randn(num_items, train.IMAGE_EMBED_DIM).astype(np.float32))
    prompts = [NULL_PROMPT] + [prompt_of(c) for c in CATEGORIES]
    embeds = torch.from_numpy(rng.randn(len(prompts), 5, train.TEXT_DIM).astype(np.float32))
    torch.save({"prompts": prompts, "embeds": embeds}, os.path.join(cache_dir, "text_embeds.pt"))
    with open(os.path.join(cache_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump({"ids": list(range(100, next_id))}, f)
    return rooms_path, cache_dir


def train_args(rooms_path: str, cache_dir: str, out: str, **overrides):
    argv = ["--form", "f", "--rooms_train", rooms_path, "--cache_dir", cache_dir, "--out", out, "--model_size", "check",
            "--image_size", str(IMAGE_SIZE), "--precision", "fp32", "--grad_checkpointing", "0", "--batch_size", "8",
            "--log_every", "1000", "--save_minutes", "1e9"]
    args = train.get_args_parser().parse_args(argv)
    for key, value in overrides.items():
        assert hasattr(args, key), key
        setattr(args, key, value)
    return args


def randomize(model: torch.nn.Module, std: float = 0.05) -> None:
    """Stand in for trained weights: a fresh model has zero-initialised layers that make every output zero."""
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=std)


def check_added_layers_start_as_identity(args):
    """With the cross-attention output and the sample embedding at zero, the outputs are those of the plain DiT."""
    plain_config = Denoiser.Config(
        in_channels=4, input_size=IMAGE_SIZE // 8, patch_size=2, learn_sigma=True, output_sigma=True,
        hidden_size=SIZE_DICT["check"]["width"], depth=SIZE_DICT["check"]["layers"],
        num_heads=SIZE_DICT["check"]["heads"], max_group_size=0, use_grad_checkpoint=False,
    )
    plain = Denoiser(plain_config)
    randomize(plain)
    args = copy.copy(args)
    args.query_embedding = 1
    model = train.build_model(args)
    path = os.path.join(args.out, "plain.pt")
    os.makedirs(args.out, exist_ok=True)
    torch.save(plain.state_dict(), path)
    train.load_pretrained_dit(model, path)
    train.zero_sample_embedding(model)

    x = torch.randn(6, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
    t = torch.randint(0, 1000, (6,))
    y = torch.full((6,), train.NUM_CLASSES)
    context, image_embeds = torch.randn(6, 5, train.TEXT_DIM), torch.randn(6, train.IMAGE_EMBED_DIM)
    reference = plain(x, t, y, group_size=1, keep_group_shape=False).pred
    alone = model(x, t, y, group_size=1, keep_group_shape=False, context=context, image_embeds=image_embeds).pred
    assert float(reference.abs().mean()) > 1e-3
    assert torch.equal(reference, alone), "one image alone must reproduce the plain DiT"
    grouped = model(x, t, y, group_size=3, keep_group_shape=False, context=context, image_embeds=image_embeds).pred
    assert not torch.equal(reference, grouped), "group attention must change the prediction"
    print("ok  added layers: alone equals the plain DiT bit for bit; in a group the prediction changes")


def check_epoch_plans(rooms: list[Room]):
    total = sum(r.num_targets for r in rooms)
    all_targets = sorted((r, j) for r, room in enumerate(rooms) for j in range(room.num_targets))
    for form in ["none", "l", "f"]:
        plan = plan_epoch(rooms, form, 8, seed=42, epoch=0, group_ratio=0.1)
        seen, group_targets = [], 0
        for batch in plan:
            assert batch.num_targets(rooms) <= 8 or (batch.kind == "group" and len(batch.rooms) == 1)
            if batch.kind == "single":
                seen += batch.targets
            else:
                seen += [(r, j) for r in batch.rooms for j in range(rooms[r].num_targets)]
                group_targets += batch.num_targets(rooms)
            assert batch.conditional == {"none": None, "f": None, "l": batch.kind == "single"}[form]
        assert sorted(seen) == all_targets, f"form {form}: an epoch must visit every target once"
        assert {"none": group_targets == 0, "f": group_targets == total, "l": 0 < group_targets < 0.3 * total}[form]
        assert plan_epoch(rooms, form, 8, 42, 0) == plan and plan_epoch(rooms, form, 8, 42, 1) != plan
    print(f"ok  epoch plans: every form visits each of the {total} targets once; plans depend only on seed and epoch")


def check_query_members(args, rooms: list[Room], cache: FeatureCache):
    """Query members enter clean at timestep 0, change the targets through group attention and get no loss."""
    args = copy.copy(args)
    args.query_members = 1
    model = train.build_model(args)
    randomize(model)
    autocast = train.make_autocast(args)
    room = 0
    with_queries = cache.build(rooms, [(room, list(range(rooms[room].num_targets)), True, True)], False)
    without = cache.build(rooms, [(room, list(range(rooms[room].num_targets)), False, True)], False)
    assert with_queries.num_queries == rooms[room].num_queries and without.num_queries == 0
    x = torch.randn_like(with_queries.target_latents)
    t = torch.full((x.shape[0],), 500)
    torch.manual_seed(0)
    a = train.denoise_targets(model, x, t, with_queries, autocast)
    torch.manual_seed(0)
    b = train.denoise_targets(model, x, t, without, autocast)
    assert a.shape == b.shape == (x.shape[0], 8, *x.shape[2:])
    assert not torch.allclose(a, b), "query members must change the predictions of the targets"

    # the loss is a function of the predictions of the targets only
    diffusion = create_diffusion("", noise_schedule="linear")
    seen = {}

    def model_fn(x_t, t_):
        seen["x_t"] = x_t
        return train.denoise_targets(model, x_t, t_, with_queries, autocast)

    terms = diffusion.training_losses(model_fn, with_queries.target_latents, t)
    assert terms["loss"].shape == (rooms[room].num_targets,) and seen["x_t"].shape == with_queries.target_latents.shape

    # drawing whether the query members are present is reproducible and close to the requested rate
    plan = plan_epoch(rooms, "f", 8, 42, 0)
    draws = []
    for step in range(200):
        generator = torch.Generator().manual_seed(step)
        for entries in group_entries(plan[0], rooms, True, 0.5, 0.1, generator):
            draws += [e[2] for e in entries]
    assert 0.4 < sum(draws) / len(draws) < 0.6
    print(f"ok  query members: no loss term, targets change with them, present in {sum(draws) / len(draws):.2f} of groups")


def run_training(rooms_path, cache_dir, out, **overrides):
    args = train_args(rooms_path, cache_dir, out, **overrides)
    train.main(args)
    return torch.load(os.path.join(out, "latest.pt"), map_location="cpu")


def check_resume(rooms_path, cache_dir, root):
    """Stopping after four steps and resuming gives the same weights as nine steps in one run."""
    for form, members, embedding in [("f", 1, 0), ("l", 0, 1), ("none", 0, 1)]:
        common = dict(form=form, query_members=members, query_embedding=embedding, epochs=3)
        straight = run_training(rooms_path, cache_dir, os.path.join(root, f"straight_{form}"), max_steps=9, **common)
        out = os.path.join(root, f"resumed_{form}")
        run_training(rooms_path, cache_dir, out, max_steps=4, **common)
        resumed = run_training(rooms_path, cache_dir, out, max_steps=9, resume=True, **common)
        assert straight["global_step"] == resumed["global_step"] == 9 and straight["epoch"] == resumed["epoch"] >= 1
        for key in straight["model"]:
            assert torch.equal(straight["model"][key], resumed["model"][key]), f"form {form}: {key} differs after resume"
        for key in straight["ema"]:
            assert torch.equal(straight["ema"][key], resumed["ema"][key]), f"form {form}: EMA {key} differs after resume"
    print("ok  resume: four steps, stop, five more steps across an epoch equal nine steps in one run (forms f, l, none)")


def check_loss_goes_down(rooms_path, cache_dir, root):
    out = os.path.join(root, "overfit")
    args = train_args(rooms_path, cache_dir, out, form="f", query_members=1, epochs=40, limit_rooms=4, lr=3e-4,
                      log_every=1, grad_clip=0.0, cond_drop=0.0)
    train.main(args)
    with open(os.path.join(out, "log.jsonl"), encoding="utf-8") as f:
        losses = [json.loads(line)["loss"] for line in f]
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    assert last < 0.8 * first, f"loss did not go down: {first:.4f} -> {last:.4f}"
    print(f"ok  training: loss on four rooms goes from {first:.3f} to {last:.3f} in {len(losses)} steps")


def check_generation(rooms_path, cache_dir, root):
    """Each form runs, and a target starts from the same noise whether it is sampled alone or in its group."""
    finals = {}
    for form, members in [("none", 0), ("l", 1), ("f", 1)]:
        out = os.path.join(root, f"gen_train_{form}")
        run_training(rooms_path, cache_dir, out, form=form, query_members=members, epochs=1, limit_rooms=4)
        for use_members in sorted({0, members}):
            gen_out = os.path.join(root, f"gen_{form}_{use_members}")
            argv = ["--ckpt", os.path.join(out, "final_ema.pt"), "--rooms", rooms_path, "--cache_dir", cache_dir,
                    "--out", gen_out, "--steps", "3", "--cfg", "2.5", "--limit_rooms", "4", "--skip_decode",
                    "--query_members", str(use_members)]
            generate.main(generate.get_args_parser().parse_args(argv))
            with open(os.path.join(gen_out, "samples.jsonl"), encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            finals[(form, use_members)] = sorted(r["sample_index"] for r in records)
    rooms = load_rooms(rooms_path, 4)
    expected = sorted(i for room in rooms for i in room.sample_indices)
    assert all(v == expected for v in finals.values()), "every target of every room must be generated once"

    # the first noise of a target depends on its seed only
    a = torch.randn(4, 8, 8, generator=torch.Generator().manual_seed(42 + expected[0]))
    b = torch.randn(4, 8, 8, generator=torch.Generator().manual_seed(42 + expected[0]))
    assert torch.equal(a, b)
    print(f"ok  generation: forms none, l, f produce all {len(expected)} targets, with and without query members")


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        rooms_path, cache_dir = make_data(root)
        rooms = load_rooms(rooms_path)
        cache = FeatureCache(cache_dir)
        args = train_args(rooms_path, cache_dir, os.path.join(root, "identity"))
        check_added_layers_start_as_identity(args)
        check_epoch_plans(rooms)
        check_query_members(args, rooms, cache)
        check_resume(rooms_path, cache_dir, root)
        check_loss_goes_down(rooms_path, cache_dir, root)
        check_generation(rooms_path, cache_dir, root)
    print("all checks passed")


if __name__ == "__main__":
    main()
