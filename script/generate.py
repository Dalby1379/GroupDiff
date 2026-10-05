"""
Generate the targets of every room with a model trained by script/train.py.

The form the model was trained with decides which predictions use group attention:

    none: conditional and unconditional predictions of every target alone
    l:    conditional prediction alone, unconditional prediction of the targets of a room together (GroupDiff-l)
    f:    both predictions of the targets of a room together (GroupDiff-f)

--query_members 1 adds the clean query images to every group prediction (only for models trained with them).
Every target has its own noise, drawn from --seed + sample_index, so a target starts from the same noise in
every form.

Example:

    python -m script.generate --ckpt work_dirs/f_embed/final_ema.pt --rooms rooms_test.json --cache_dir cache \
        --cfg 3.5 --out work_dirs/f_embed/test_cfg3.5
"""

import argparse
import json
import logging
import os
from argparse import Namespace

import numpy as np
import torch
from PIL import Image

from diffusion import create_diffusion

from .data import FeatureCache, GroupInputs, Room, load_rooms
from .train import build_model, denoise_targets, make_autocast

logger = logging.getLogger("GroupDiff")


def load_model(path: str, weights: str, device):
    """Rebuild the model from the arguments stored with the weights."""
    state = torch.load(path, map_location="cpu")
    train_args = Namespace(**state["args"])
    model = build_model(train_args)
    model.load_state_dict(state["ema"] if weights == "ema" else state["model"])
    model.use_grad_checkpoint = False
    return model.to(device).eval(), train_args


class GuidedDenoiser:
    """Classifier-free guidance in which each prediction is made alone or in the group, as the form prescribes."""

    def __init__(self, model, form: str, cfg: float, cond_group: GroupInputs, uncond_group: GroupInputs,
                 cond_single: GroupInputs, uncond_single: GroupInputs, autocast):
        self.model, self.cfg, self.autocast = model, cfg, autocast
        # the "single" inputs hold one group per target, in the same order of targets as the group inputs
        self.cond = cond_group if form == "f" else cond_single
        self.uncond = uncond_single if form == "none" else uncond_group

    def __call__(self, x, t):
        cond = denoise_targets(self.model, x, t, self.cond, self.autocast)
        if self.cfg == 1.0:
            return cond
        uncond = denoise_targets(self.model, x, t, self.uncond, self.autocast)
        channels = x.shape[1]
        eps = uncond[:, :channels] + self.cfg * (cond[:, :channels] - uncond[:, :channels])
        # the learned variance is taken from the conditional prediction, as in models/dit.py
        return torch.cat([eps, cond[:, channels:]], dim=1)


def room_inputs(cache: FeatureCache, rooms: list[Room], indices: list[int], with_queries: bool, conditional: bool,
                single: bool, use_image_embeds: bool) -> GroupInputs:
    """Inputs of rooms of one shape: the targets of a room as one group, or every target as its own group."""
    if single:
        entries = [(r, [j], False, conditional) for r in indices for j in range(rooms[r].num_targets)]
    else:
        entries = [(r, list(range(rooms[r].num_targets)), with_queries, conditional) for r in indices]
    return cache.build(rooms, entries, use_image_embeds)


@torch.no_grad()
def sample(model, diffusion, cache, rooms, indices, args, train_args, autocast) -> tuple[torch.Tensor, GroupInputs]:
    """Sample the targets of rooms that have the same numbers of targets and of query items."""
    use_embeds = bool(train_args.query_embedding)
    with_queries = bool(args.query_members)
    cond_group = room_inputs(cache, rooms, indices, with_queries, True, False, use_embeds)
    guided = GuidedDenoiser(
        model,
        train_args.form,
        args.cfg,
        cond_group,
        room_inputs(cache, rooms, indices, with_queries, False, False, use_embeds),
        room_inputs(cache, rooms, indices, False, True, True, use_embeds),
        room_inputs(cache, rooms, indices, False, False, True, use_embeds),
        autocast,
    )
    shape = cond_group.target_latents.shape[1:]
    generators = [torch.Generator().manual_seed(args.seed + i) for i in cond_group.sample_indices]

    def noise():
        return torch.stack([torch.randn(shape, generator=g) for g in generators]).to(cache.device)

    x = noise()
    for i in reversed(range(diffusion.num_timesteps)):
        t = torch.full((x.shape[0],), i, dtype=torch.long, device=cache.device)
        out = diffusion.p_mean_variance(guided, x, t, clip_denoised=False)
        x = out["mean"]
        if i > 0:
            x = x + torch.exp(0.5 * out["log_variance"]) * noise()
    return x, cond_group


@torch.no_grad()
def decode(vae, latents: torch.Tensor) -> list[Image.Image]:
    images = vae.decode(latents / vae.config.scaling_factor).sample
    images = (images / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy() * 255.0
    return [Image.fromarray(im) for im in images.astype(np.uint8)]


def main(args):
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_args = load_model(args.ckpt, args.weights, device)
    if args.query_members and not train_args.query_members:
        raise ValueError("this model was trained without query members")
    if args.precision:
        train_args.precision = args.precision
    autocast = make_autocast(train_args)
    diffusion = create_diffusion(str(args.steps), noise_schedule="linear")
    cache = FeatureCache(args.cache_dir, device)
    rooms = load_rooms(args.rooms, args.limit_rooms)

    vae = None
    if not args.skip_decode:
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()

    record_path = os.path.join(args.out, "samples.jsonl")
    done = set()
    if os.path.isfile(record_path):
        with open(record_path, encoding="utf-8") as f:
            done = {json.loads(line)["sample_index"] for line in f}
    todo = [r for r, room in enumerate(rooms) if not set(room.sample_indices) <= done]
    logger.info("rooms: %d, to generate: %d, form %s, cfg %.2f, query members %d", len(rooms), len(todo),
                train_args.form, args.cfg, args.query_members)

    # rooms of one shape are sampled together
    by_shape: dict[tuple[int, int], list[int]] = {}
    for r in todo:
        by_shape.setdefault((rooms[r].num_targets, rooms[r].num_queries), []).append(r)
    for shape in sorted(by_shape):
        indices = by_shape[shape]
        per_batch = max(1, args.batch_size // shape[0])
        for start in range(0, len(indices), per_batch):
            chunk = indices[start : start + per_batch]
            latents, inputs = sample(model, diffusion, cache, rooms, chunk, args, train_args, autocast)
            images = decode(vae, latents) if vae is not None else [None] * latents.shape[0]
            with open(record_path, "a", encoding="utf-8") as f:
                for k, image in enumerate(images):
                    name = f"{inputs.sample_indices[k]:05d}_{inputs.scene_ids[k]}_{inputs.target_ids[k]}.png"
                    if image is not None:
                        image.save(os.path.join(args.out, "images", name))
                    record = {
                        "sample_index": inputs.sample_indices[k],
                        "scene_id": inputs.scene_ids[k],
                        "target_id": inputs.target_ids[k],
                        "image": os.path.join("images", name),
                        "seed": args.seed + inputs.sample_indices[k],
                    }
                    f.write(json.dumps(record) + "\n")
            logger.info("targets %d, queries %d: %d / %d rooms", shape[0], shape[1], start + len(chunk), len(indices))

    settings = {**vars(args), "form": train_args.form, "trained_with": vars(train_args)}
    with open(os.path.join(args.out, "generate_args.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)


def get_args_parser():
    parser = argparse.ArgumentParser("Generate the targets of rooms")
    parser.add_argument("--ckpt", required=True, type=str, help="final_ema.pt or latest.pt of script/train.py")
    parser.add_argument("--weights", default="ema", type=str, choices=["ema", "model"])
    parser.add_argument("--rooms", required=True, type=str)
    parser.add_argument("--cache_dir", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--query_members", default=0, type=int, choices=[0, 1], help="clean query images in the groups")
    parser.add_argument("--cfg", default=1.5, type=float, help="guidance weight w of eps_u + w (eps_c - eps_u)")
    parser.add_argument("--steps", default=250, type=int, help="sampling steps of the DDPM sampler")
    parser.add_argument("--seed", default=42, type=int, help="the noise of a target is drawn from seed + sample_index")
    parser.add_argument("--batch_size", default=64, type=int, help="targets sampled together")
    parser.add_argument("--limit_rooms", default=0, type=int)
    parser.add_argument("--precision", default="", type=str, choices=["", "bf16", "fp32"])
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse", type=str)
    parser.add_argument("--skip_decode", action="store_true", help="do not decode images (for tests)")
    return parser


if __name__ == "__main__":
    main(get_args_parser().parse_args())
