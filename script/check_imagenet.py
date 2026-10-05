"""
Small checks of script/generate_imagenet.py that run on a CPU in a few seconds, on a small random model.

    python -m script.check_imagenet

They cover what generation with one class per image relies on: an image gets its own class and its own noise
whatever groups it is sampled with, the images of a group interact only through the unconditional prediction,
and the weights that are loaded are the ones asked for.
"""

import json
import os
import tempfile

import torch

from models.dit import DiT, DiT_models
from models.model_utils import SIZE_DICT

from . import generate_imagenet as generate
from . import imagenet_labels

IMAGE_SIZE = 64  # latents of 8 x 8, so that every check stays fast
SIZE_DICT["check"] = {"width": 64, "layers": 2, "heads": 4}  # a model small enough for a CPU
DiT_models["DiT_check"] = lambda **kwargs: DiT(model_size="check", **kwargs)


def make_args(root: str, groups: list[dict], **overrides):
    path = os.path.join(root, f"groups{len(os.listdir(root))}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"groups": groups}, f)
    argv = ["--ckpt", os.path.join(root, "weights.pth"), "--groups", path, "--out", os.path.join(root, "out"),
            "--cfg", "2.0", "--steps", "3", "--precision", "fp32", "--model", "DiT_check",
            "--img_size", str(IMAGE_SIZE), "--skip_decode"]
    args = generate.get_args_parser().parse_args(argv)
    for key, value in overrides.items():
        assert hasattr(args, key), key
        setattr(args, key, value)
    return args


def make_weights(root: str):
    """Stand in for the released weights: a fresh model has zero-initialised layers that make every output zero."""
    args = make_args(root, [])
    model = generate.build_model(args)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.05)
    ema = {name: p.detach() * 0.5 for name, p in model.named_parameters()}
    torch.save({"model": model.state_dict(), "model_ema": ema}, args.ckpt)


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal up to the rounding that the size of a batch changes, relative to the scale of the latents."""
    return float((a - b).abs().max()) <= 1e-5 * float(b.abs().max())


def fix_sample_ids(model):
    """The released code draws the ids of the sample embedding anew in every call. Fix them to compare runs."""
    model.denoiser._generate_random_sample_ids = lambda groups, size, device: (
        torch.arange(size, device=device).unsqueeze(0).repeat(groups, 1)
    )


def run(model, args, groups: list[dict]) -> dict[tuple[str, str], torch.Tensor]:
    """Latents of every image, by group and member. Groups of one size are sampled together, as in main()."""
    path = os.path.join(os.path.dirname(args.groups), "run.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"groups": groups}, f)
    loaded = generate.load_groups(path)
    latents = {}
    for size in sorted({len(g.labels) for g in loaded}):
        same = [g for g in loaded if len(g.labels) == size]
        x = generate.sample(model, model.gen_diffusion, same, args, torch.device("cpu"))
        for n, (g, k) in enumerate((g, k) for g in same for k in range(size)):
            latents[(g.name, g.members[k])] = x[n]
    return latents


def check_weights(root: str):
    args = make_args(root, [])
    ema = generate.load_model(args, torch.device("cpu"))
    args.weights = "model"
    plain = generate.load_model(args, torch.device("cpu"))
    ratio = float(ema.denoiser.final_layer.linear.weight.norm() / plain.denoiser.final_layer.linear.weight.norm())
    assert abs(ratio - 0.5) < 1e-6, "--weights ema must load the averaged weights"
    args.ckpt = os.path.join(root, "missing.pth")
    try:
        generate.load_model(args, torch.device("cpu"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing weight file must be an error")
    print("ok  weights: ema and model load different weights; a missing file is an error")


def check_images_are_independent_of_batching(model, args):
    groups = [
        {"name": "a", "labels": [1, 2]},
        {"name": "b", "labels": [3, 4, 5, 6]},
        {"name": "c", "labels": [7, 8]},
        {"name": "d", "labels": [9]},
    ]
    together = run(model, args, groups)
    assert len(together) == 9
    assert float(together[("a", "0")].abs().mean()) > 1e-3
    # one group at a time; the other groups stay in the file so that the sample indices, and so the noise, are kept
    path = os.path.join(os.path.dirname(args.groups), "run.json")
    for g in generate.load_groups(path):
        x = generate.sample(model, model.gen_diffusion, [g], args, torch.device("cpu"))
        for k, member in enumerate(g.members):
            assert close(x[k], together[(g.name, member)]), "an image must not depend on the batch"
    torch.manual_seed(123)  # whatever the state of the global generator before the call
    assert all(torch.equal(v, together[k]) for k, v in run(model, args, groups).items()), "reruns must be identical"
    print("ok  batching: 9 images in groups of 1, 2 and 4 are the same sampled together or group by group; reruns are identical")


def check_group_interaction(model, args):
    base = [{"name": "g", "labels": [10, 20, 30]}, {"name": "h", "labels": [40, 50, 60]}]
    other_label = [{"name": "g", "labels": [10, 21, 30]}, {"name": "h", "labels": [40, 50, 60]}]
    own_label = [{"name": "g", "labels": [11, 20, 30]}, {"name": "h", "labels": [40, 50, 60]}]
    first = ("g", "0")

    args.group_attention = 0
    alone = run(model, args, base)
    assert torch.equal(alone[first], run(model, args, other_label)[first]), "alone, the other classes must not matter"
    assert not torch.equal(alone[first], run(model, args, own_label)[first]), "the class of an image must matter"
    single = run(model, args, [{"name": "g", "labels": [10]}, {"name": "pad", "labels": [0, 0]},
                               {"name": "h", "labels": [40, 50, 60]}])
    assert close(alone[first], single[first]), "alone, an image equals the image sampled by itself"

    args.group_attention = 1
    grouped = run(model, args, base)
    assert not torch.equal(grouped[first], run(model, args, own_label)[first])
    # in a group the class of another image changes this image, through that image's latent, so not within one step
    assert not torch.equal(grouped[first], run(model, args, other_label)[first]), "the images of a group must interact"
    diffusion = model.gen_diffusion
    x = torch.randn(6, model.token_channels, model.input_size, model.input_size)
    t = torch.full((6,), diffusion.num_timesteps - 1)
    means = [
        diffusion.p_mean_variance(generate.GuidedPrediction(model, torch.tensor(labels), 3, args), x, t,
                                  clip_denoised=False)["mean"]
        for labels in ([10, 20, 30, 40, 50, 60], [10, 21, 30, 40, 50, 60])
    ]
    assert torch.equal(means[0][0], means[1][0]), "the classes must enter the conditional prediction only"
    assert not torch.equal(means[0][1], means[1][1])

    args.cfg = 1.0
    assert close(run(model, args, base)[first], run(model, args, single_groups(base))[first]), \
        "with a guidance weight of 1 there is no unconditional prediction and no interaction"
    args.cfg = 2.0
    print("ok  groups: alone, an image depends on its own class only; in a group the other images change it, "
          "their classes only through their latents; a guidance weight of 1 removes the interaction")


def single_groups(groups: list[dict]) -> list[dict]:
    """Every image as its own group, with the names and the order kept so that the sample indices stay the same."""
    return [{"name": g["name"], "labels": [label], "members": [str(k)]} for g in groups for k, label in
            enumerate(g["labels"])]


def check_main(root: str):
    groups = [{"name": "room0", "labels": [1, 2, 3]}, {"name": "room1", "labels": [4, 5], "members": ["x", "y"]}]
    args = make_args(root, groups)
    generate.main(args)
    with open(os.path.join(args.out, "samples.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert sorted(r["sample_index"] for r in records) == [0, 1, 2, 3, 4]
    assert {(r["group"], r["member"], r["label"]) for r in records} == {
        ("room0", "0", 1), ("room0", "1", 2), ("room0", "2", 3), ("room1", "x", 4), ("room1", "y", 5)}
    generate.main(args)
    with open(os.path.join(args.out, "samples.jsonl"), encoding="utf-8") as f:
        assert len(f.readlines()) == 5, "a second run must not generate the finished images again"

    for bad, message in [([{"name": "big", "labels": [1, 2, 3, 4, 5]}], "a group above the sample embedding"),
                         ([{"name": "label", "labels": [1000]}], "a class outside the label embedding")]:
        try:
            generate.main(make_args(root, bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{message} must be an error")
    print("ok  main: every image is written once with its group, member and class; a rerun resumes; "
          "groups above the sample embedding and classes outside the label embedding are errors")


def check_group_files(root: str):
    classes = {"sofa": 831, "table": 532, "chair#stool": 559}
    rooms = {"split": "check", "rooms": [
        {"scene_id": "r0", "queries": [], "targets": [{"id": 1, "category": "sofa", "sample_index": 0},
                                                      {"id": 2, "category": "chair#stool", "sample_index": 1}]},
        {"scene_id": "r1", "queries": [], "targets": [{"id": k, "category": "table", "sample_index": k}
                                                      for k in range(2, 7)]},
    ]}
    classes_path, rooms_path = os.path.join(root, "classes.json"), os.path.join(root, "rooms.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f)
    with open(rooms_path, "w", encoding="utf-8") as f:
        json.dump(rooms, f)
    parser = imagenet_labels.get_args_parser()
    out = os.path.join(root, "groups_rooms.json")
    imagenet_labels.groups(parser.parse_args(["groups", "--classes", classes_path, "--rooms", rooms_path, "--out", out]))
    loaded = generate.load_groups(out)
    assert [(g.name, g.labels, g.members) for g in loaded] == [("r0", [831, 559], ["1-sofa", "2-chair+stool"])], \
        "a room becomes one group with the class of every target; a room above the group size is left out"
    out = os.path.join(root, "groups_classes.json")
    imagenet_labels.groups(parser.parse_args(["groups", "--classes", classes_path, "--out", out,
                                              "--groups_per_category", "2", "--group_size", "3"]))
    loaded = generate.load_groups(out)
    assert len(loaded) == 6 and all(len(set(g.labels)) == 1 and len(g.labels) == 3 for g in loaded)
    print("ok  group files: rooms give one group per room with one class per target; without rooms, groups of one class")


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        make_weights(root)
        check_weights(root)
        args = make_args(root, [])
        model = generate.load_model(args, torch.device("cpu"))
        fix_sample_ids(model)
        check_images_are_independent_of_batching(model, args)
        check_group_interaction(model, args)
        check_main(root)
        check_group_files(root)
    print("all checks passed")


if __name__ == "__main__":
    main()
