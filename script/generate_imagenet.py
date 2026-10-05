"""
Generate images with the released class-conditional GroupDiff weights, one ImageNet-1k class per image.

eval.py gives every image of a group the same class and cycles through the 1000 classes. Here the classes are
read from a file, one per image, so that a group can hold different classes, such as the furniture items of a
room after their categories have been expressed as ImageNet-1k classes (script/imagenet_labels.py).

Group file (JSON):

    {"groups": [{"name": "room0", "labels": [831, 532, 846], "members": ["sofa", "table", "lamp"]}, ...]}

`members` is optional and only names the images. The images of the file are numbered in order; this number is
the sample index.

The released weights are GroupDiff-l: the conditional prediction is made for each image alone, and the images of
a group attend to each other only in the unconditional prediction. The class of an image therefore enters its
own conditional prediction and nothing else. A group can hold at most as many images as the sample embedding of
the weights has rows (4 for the released weights).

    python -m script.generate_imagenet --ckpt released_model/gdiff-l-4-dit-xl-2-resume.pth \
        --groups groups.json --cfg 1.65 --out work_dirs/imagenet_furniture

`--group_attention 0` makes the unconditional prediction for each image alone as well. The noise of an image is
drawn from --seed + sample index, so an image starts from the same noise in both settings.

The released code draws the ids of the sample embedding at random in every call of the denoiser, so the ids of an
image change from step to step and depend on the other images of the batch (`--sample_ids random`, the default).
`--sample_ids position` gives an image the id of its position in its group in every call instead; an image then
depends only on its group, and the two settings of --group_attention differ in the attention alone.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from models.dit import DiT_models
from models.ema import SimpleEMAModel

logger = logging.getLogger("GroupDiff")


@dataclass
class Group:
    name: str
    labels: list[int]
    members: list[str]
    sample_indices: list[int]


def load_groups(path: str, limit: int = 0) -> list[Group]:
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)["groups"]
    groups, next_index = [], 0
    for entry in entries:
        labels = [int(label) for label in entry["labels"]]
        members = [str(m) for m in entry.get("members", range(len(labels)))]
        if len(members) != len(labels):
            raise ValueError(f"group {entry['name']}: {len(labels)} labels but {len(members)} members")
        groups.append(Group(str(entry["name"]), labels, members, list(range(next_index, next_index + len(labels)))))
        next_index += len(labels)
    return groups[:limit] if limit else groups


def build_model(args):
    return DiT_models[args.model](
        img_size=args.img_size,
        patch_size=args.patch_size,
        tokenizer_patch_size=args.tokenizer_patch_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes,
        num_sampling_steps=str(args.steps),
        num_max_sample=args.num_max_sample,
        noise_schedule="linear",
    )


def load_model(args, device):
    """Build the model and load the released weights; a missing or mismatching file is an error."""
    model = build_model(args)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if args.weights == "ema":
        ema = SimpleEMAModel(model)
        ema.load_state_dict(checkpoint["model_ema"])
        ema.copy_to(model)
    total = sum(float(p.double().abs().sum()) for p in model.parameters())
    logger.info("loaded %s weights of %s (sum of absolute values %.6e)", args.weights, args.ckpt, total)
    return model.to(device).eval()


class GuidedPrediction:
    """Classifier-free guidance of models/dit.py, with the labels of the images held fixed over the steps."""

    def __init__(self, model, labels: torch.Tensor, group_size: int, args):
        self.model, self.labels, self.cfg = model, labels, args.cfg
        self.null = torch.full_like(labels, model.num_classes)
        self.group_size = group_size if args.group_attention else 1
        # models/dit.py compares these bounds with timesteps in [0, 1000)
        self.low, self.high = args.guidance_low * 1000.0, args.guidance_high * 1000.0

    def __call__(self, x, t):
        if self.cfg == 1.0:
            return self.model.net(x, t, self.labels)
        out = self.model.forward_with_cfg(
            torch.cat([x, x]),
            torch.cat([t, t]),
            torch.cat([self.labels, self.null]),
            self.cfg,
            cond_group_size=1,
            uncond_group_size=self.group_size,
            guidance_low=self.low,
            guidance_high=self.high,
        )
        return out[: x.shape[0]]


def use_position_ids(model, positions: torch.Tensor):
    """Give every image the id of its position in its group, alone or in the group, in place of the random ids."""
    model.denoiser._generate_random_sample_ids = lambda groups, size, device: positions.view(groups, size)


@torch.no_grad()
def sample(model, diffusion, groups: list[Group], args, device) -> torch.Tensor:
    """Sample groups of one size. The images of a group are consecutive, as the group attention expects."""
    size = len(groups[0].labels)
    assert all(len(g.labels) == size for g in groups)
    labels = torch.tensor([label for g in groups for label in g.labels], device=device)
    indices = [i for g in groups for i in g.sample_indices]
    predict = GuidedPrediction(model, labels, size, args)
    if args.sample_ids == "position":
        use_position_ids(model, torch.arange(size, device=device).repeat(len(groups)))
    shape = (model.token_channels, model.input_size, model.input_size)
    generators = [torch.Generator().manual_seed(args.seed + i) for i in indices]
    # random ids of the sample embedding come from the global generator; pin it so that a rerun gives the same images
    torch.manual_seed(args.seed + min(indices))

    def noise():
        return torch.stack([torch.randn(shape, generator=g) for g in generators]).to(device)

    x = noise()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        for i in reversed(range(diffusion.num_timesteps)):
            t = torch.full((x.shape[0],), i, dtype=torch.long, device=device)
            out = diffusion.p_mean_variance(predict, x, t, clip_denoised=False)
            x = out["mean"].float()
            if i > 0:
                x = x + torch.exp(0.5 * out["log_variance"].float()) * noise()
    return x


@torch.no_grad()
def decode(vae, latents: torch.Tensor) -> list[Image.Image]:
    images = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
    images = (images / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy() * 255.0
    return [Image.fromarray(im) for im in images.astype(np.uint8)]


def main(args):
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = load_groups(args.groups, args.limit_groups)
    too_large = [g.name for g in groups if len(g.labels) > args.num_max_sample]
    if too_large and (args.group_attention or args.sample_ids == "position"):
        raise ValueError(
            f"{len(too_large)} groups hold more than {args.num_max_sample} images, the rows of the sample embedding "
            f"(first: {too_large[0]})"
        )
    out_of_range = [g.name for g in groups if not all(0 <= label < args.num_classes for label in g.labels)]
    if out_of_range:
        raise ValueError(f"labels outside [0, {args.num_classes}) in {len(out_of_range)} groups ({out_of_range[0]})")
    if args.cfg < 1.0:
        raise ValueError("--cfg is the weight w of eps_u + w (eps_c - eps_u) and must be at least 1")
    if args.cfg == 1.0 and args.group_attention:
        logger.warning("--cfg 1.0 uses the conditional prediction only: the images of a group do not interact")

    model = load_model(args, device)
    if args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)

    vae = None
    if not args.skip_decode:
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()

    record_path = os.path.join(args.out, "samples.jsonl")
    done = set()
    if os.path.isfile(record_path):
        with open(record_path, encoding="utf-8") as f:
            done = {json.loads(line)["sample_index"] for line in f}
    todo = [g for g in groups if not set(g.sample_indices) <= done]
    logger.info("groups: %d, to generate: %d, cfg %.3f, group attention %d", len(groups), len(todo), args.cfg,
                args.group_attention)

    # groups of one size are sampled together
    by_size: dict[int, list[Group]] = {}
    for g in todo:
        by_size.setdefault(len(g.labels), []).append(g)
    for size in sorted(by_size):
        same = by_size[size]
        per_batch = max(1, args.batch_size // size)
        for start in range(0, len(same), per_batch):
            chunk = same[start : start + per_batch]
            latents = sample(model, model.gen_diffusion, chunk, args, device)
            images = decode(vae, latents) if vae is not None else [None] * latents.shape[0]
            members = [(g, k) for g in chunk for k in range(size)]
            with open(record_path, "a", encoding="utf-8") as f:
                for (g, k), image in zip(members, images):
                    name = f"{g.sample_indices[k]:05d}_{g.name}_{g.members[k]}_class-{g.labels[k]:04d}.png"
                    if image is not None:
                        image.save(os.path.join(args.out, "images", name))
                    record = {
                        "sample_index": g.sample_indices[k],
                        "group": g.name,
                        "member": g.members[k],
                        "label": g.labels[k],
                        "image": os.path.join("images", name),
                        "seed": args.seed + g.sample_indices[k],
                    }
                    f.write(json.dumps(record) + "\n")
            logger.info("groups of %d: %d / %d", size, start + len(chunk), len(same))

    with open(os.path.join(args.out, "generate_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=1)


def get_args_parser():
    parser = argparse.ArgumentParser("Generate images with one ImageNet-1k class per image")
    parser.add_argument("--ckpt", required=True, type=str, help="released weights, e.g. gdiff-l-4-dit-xl-2-resume.pth")
    parser.add_argument("--weights", default="ema", type=str, choices=["ema", "model"])
    parser.add_argument("--groups", required=True, type=str, help="group file, see the top of this file")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--cfg", required=True, type=float, help="guidance weight w of eps_u + w (eps_c - eps_u)")
    parser.add_argument("--group_attention", default=1, type=int, choices=[0, 1],
                        help="1: unconditional prediction of the images of a group together; 0: each image alone")
    parser.add_argument("--sample_ids", default="random", type=str, choices=["random", "position"],
                        help="ids of the sample embedding: drawn in every call as released, or the position in the group")
    parser.add_argument("--guidance_low", default=0.0, type=float, help="guidance is applied for t / 1000 in [low, high]")
    parser.add_argument("--guidance_high", default=1.0, type=float)
    parser.add_argument("--steps", default=250, type=int, help="sampling steps of the DDPM sampler")
    parser.add_argument("--seed", default=0, type=int, help="the noise of an image is drawn from seed + sample index")
    parser.add_argument("--batch_size", default=64, type=int, help="images sampled together")
    parser.add_argument("--limit_groups", default=0, type=int)
    parser.add_argument("--precision", default="bf16", type=str, choices=["bf16", "fp32"])
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse", type=str)
    parser.add_argument("--skip_decode", action="store_true", help="do not decode images (for tests)")

    # the model of the released weights
    parser.add_argument("--model", default="DiT_xl", type=str, choices=sorted(DiT_models))
    parser.add_argument("--patch_size", default=2, type=int)
    parser.add_argument("--num_max_sample", default=4, type=int, help="rows of the sample embedding")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--token_channels", default=4, type=int)
    parser.add_argument("--tokenizer_patch_size", default=8, type=int)
    parser.add_argument("--num_classes", default=1000, type=int)
    return parser


if __name__ == "__main__":
    main(get_args_parser().parse_args())
