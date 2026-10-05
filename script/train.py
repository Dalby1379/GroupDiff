"""
Group training of a text-conditional DiT on rooms of furniture items.

The targets of a room are the images to denoise. --form selects how they are trained:

    none: every target alone (the baseline without group attention)
    l:    GroupDiff-l. Unconditional batches (group_ratio of the data) denoise the targets of a room together,
          conditional batches denoise every target alone.
    f:    GroupDiff-f. Every batch denoises the targets of a room together.

The query items of a room condition its targets in up to two ways:

    --query_embedding 1: the mean pooled image embedding of the query items is given as image tokens
    --query_members 1:   the clean query images join the group at timestep 0 and carry no loss

Example:

    python -m script.train --form f --query_embedding 1 --rooms_train rooms_train.json --rooms_val rooms_val.json \
        --cache_dir cache --init_ckpt DiT-XL-2-256x256.pt --out work_dirs/f_embed
"""

import argparse
import json
import logging
import math
import os
import threading
import time

import torch

from diffusion import create_diffusion
from models.denoiser import Denoiser
from models.ema import SimpleEMAModel
from models.model_utils import SIZE_DICT

from .data import Batch, FeatureCache, GroupInputs, Room, group_entries, load_rooms, plan_epoch

logger = logging.getLogger("GroupDiff")

NUM_CLASSES = 1000  # the class label is fixed to the null class of the ImageNet model
TEXT_DIM = 768  # CLIP ViT-L/14
IMAGE_EMBED_DIM = 1280  # OpenCLIP ViT-bigG/14


#################################################################################
#                                     Model                                     #
#################################################################################


def build_model(args) -> Denoiser:
    size = SIZE_DICT[args.model_size]
    config = Denoiser.Config(
        in_channels=4,
        input_size=args.image_size // 8,
        patch_size=2,
        hidden_size=size["width"],
        depth=size["layers"],
        num_heads=size["heads"],
        num_classes=NUM_CLASSES,
        learn_sigma=True,
        output_sigma=True,
        max_group_size=args.max_group_size,
        use_grad_checkpoint=bool(args.grad_checkpointing),
        context_dim=TEXT_DIM,
        image_embed_dim=IMAGE_EMBED_DIM if args.query_embedding else 0,
    )
    return Denoiser(config)


def load_pretrained_dit(model: Denoiser, path: str) -> None:
    """
    Start from the class-conditional DiT weights of https://github.com/facebookresearch/DiT.
    Only the layers that this model adds may be missing from the file.
    """
    state = torch.load(path, map_location="cpu")
    state = state.get("ema", state) if isinstance(state, dict) and "ema" in state else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    added = ("context_proj.", "image_proj.", "image_norm.", "sample_embedder.")
    not_added = [k for k in missing if not k.startswith(added) and ".cross_attn." not in k and ".norm_cross." not in k]
    assert not unexpected, f"unexpected keys in {path}: {unexpected[:5]}"
    assert not not_added, f"keys of the DiT missing from {path}: {not_added[:5]}"
    logger.info("loaded %s (%d tensors, %d new tensors keep their initialisation)", path, len(state), len(missing))


def zero_sample_embedding(model: Denoiser) -> None:
    """With the sample embedding at zero, the model starts with exactly the outputs of the weights it was loaded from."""
    if model.sample_embedder is not None:
        torch.nn.init.zeros_(model.sample_embedder.embedding_table.weight)


def denoise_targets(model: Denoiser, x_t: torch.Tensor, t: torch.Tensor, inputs: GroupInputs, autocast) -> torch.Tensor:
    """
    Predict for the targets of every group. The query members, if any, are appended clean at timestep 0, and
    only the predictions of the targets are returned.
    x_t: (R * num_targets, C, H, W) noisy targets, t: (R * num_targets,) their timesteps.
    """
    n_t, n_q = inputs.num_targets, inputs.num_queries
    num_groups = x_t.shape[0] // n_t
    x = x_t.view(num_groups, n_t, *x_t.shape[1:])
    t = t.view(num_groups, n_t)
    if n_q:
        x = torch.cat([x, inputs.query_latents.to(x.dtype)], dim=1)
        t = torch.cat([t, torch.zeros(num_groups, n_q, dtype=t.dtype, device=t.device)], dim=1)
    y = torch.full_like(t, NUM_CLASSES)
    with autocast():
        pred = model(x, t, y, context=inputs.text, image_embeds=inputs.image_embeds).pred
    return pred[:, :n_t].reshape(num_groups * n_t, *pred.shape[2:]).float()


def make_autocast(args):
    enabled = args.precision == "bf16"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return lambda: torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled)


#################################################################################
#                                     Loss                                      #
#################################################################################


def split_groups(entries: list, rooms: list[Room], chunk_images: int) -> list[list]:
    """Split groups of one shape so that one forward pass sees at most chunk_images images (0: no limit)."""
    if chunk_images <= 0:
        return [entries]
    r, target_idx, with_queries, _ = entries[0]
    group_size = len(target_idx) + (rooms[r].num_queries if with_queries else 0)
    per_chunk = max(1, chunk_images // group_size)
    return [entries[i : i + per_chunk] for i in range(0, len(entries), per_chunk)]


def batch_loss(model, diffusion, cache: FeatureCache, rooms: list[Room], batch: Batch, args, step_seed: int, autocast,
               backward: bool):
    """
    Loss of one batch, averaged over its targets. Groups of different shapes go through the model one shape at a
    time; with backward=True their gradients accumulate, so the caller makes a single optimizer step.
    """
    generator = torch.Generator().manual_seed(step_seed)
    torch.manual_seed(step_seed)
    num_targets = batch.num_targets(rooms)
    total, images = 0.0, 0
    for entries in group_entries(batch, rooms, bool(args.query_members), args.query_member_drop, args.cond_drop, generator):
        for chunk in split_groups(entries, rooms, args.chunk_images):
            inputs = cache.build(rooms, chunk, bool(args.query_embedding), random_flip=bool(args.random_flip))
            n_t, n_q, num_groups = inputs.num_targets, inputs.num_queries, inputs.num_groups
            if batch.kind == "group":
                # the targets of a group share one timestep
                t = torch.randint(0, diffusion.num_timesteps, (num_groups,), device=cache.device).repeat_interleave(n_t)
            else:
                t = torch.randint(0, diffusion.num_timesteps, (num_groups * n_t,), device=cache.device)
            terms = diffusion.training_losses(
                lambda x_t, t_, inputs=inputs: denoise_targets(model, x_t, t_, inputs, autocast),
                inputs.target_latents,
                t,
            )
            loss = terms["loss"].sum() / num_targets
            if backward:
                loss.backward()
            total += float(loss.detach())
            images += num_groups * (n_t + n_q)
    return total, num_targets, images


#################################################################################
#                                  Checkpoints                                  #
#################################################################################


class HubSync:
    """Keeps the latest checkpoint of a run on a branch of a Hugging Face repository."""

    def __init__(self, repo: str, branch: str):
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo, self.branch = repo, branch
        self.api.create_repo(repo, private=True, exist_ok=True)
        self.api.create_branch(repo, branch=branch, exist_ok=True)
        self.thread = None

    def _upload(self, path: str, name: str):
        try:
            self.api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=self.repo, revision=self.branch)
            # keep a single commit on the branch, so that replaced checkpoints do not stay in the history
            self.api.super_squash_history(self.repo, branch=self.branch)
            logger.info("uploaded %s to %s@%s", name, self.repo, self.branch)
        except Exception as e:  # an upload that fails must not stop training
            logger.warning("upload of %s failed: %s", name, e)
        finally:
            if path.endswith(".uploading"):
                os.remove(path)

    def upload(self, path: str, name: str, wait: bool = False):
        """Upload in the background. A request made while the previous upload is still running is skipped."""
        if self.thread is not None and self.thread.is_alive():
            if not wait:
                return
            self.thread.join()
        link = path + ".uploading"
        if os.path.exists(link):
            os.remove(link)
        os.link(path, link)  # the file at `path` may be replaced while the upload reads it
        self.thread = threading.Thread(target=self._upload, args=(link, name), daemon=True)
        self.thread.start()
        if wait:
            self.thread.join()

    def download(self, name: str, path: str) -> bool:
        from huggingface_hub import hf_hub_download

        try:
            src = hf_hub_download(self.repo, name, revision=self.branch, local_dir=os.path.dirname(path))
        except Exception:
            return False
        if os.path.abspath(src) != os.path.abspath(path):
            os.replace(src, path)
        return True


def save_checkpoint(path: str, model, ema, optimizer, epoch: int, batch_index: int, global_step: int, args):
    state = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch_index": batch_index,
        "global_step": global_step,
        "args": vars(args),
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def create_optimizer(model, args):
    # the same split as utils/builders.py: no weight decay on biases, norms and embeddings
    def exclude(name, p):
        keywords = ["ln", "bias", "embedding", "norm", "gamma", "embed", "token", "diffloss"]
        return p.ndim < 2 or any(k in name for k in keywords)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": [p for n, p in named if exclude(n, p)], "weight_decay": 0.0},
            {"params": [p for n, p in named if not exclude(n, p)], "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )


#################################################################################
#                                   Training                                    #
#################################################################################


def step_seed(seed: int, global_step: int) -> int:
    return seed * 1_000_003 + global_step


@torch.no_grad()
def validate(model, diffusion, cache, rooms: list[Room], args, autocast) -> dict:
    """Loss of the training objective of this form on the validation rooms, with fixed noise."""
    model.eval()
    grad_checkpoint, model.use_grad_checkpoint = model.use_grad_checkpoint, False
    sums, counts = {}, {}
    plan = plan_epoch(rooms, args.form, args.batch_size, args.seed, -1, args.group_ratio)
    for i, batch in enumerate(plan):
        loss, n, _ = batch_loss(model, diffusion, cache, rooms, batch, args, step_seed(args.seed + 1, i), autocast, False)
        for key in (batch.kind, "all"):
            sums[key] = sums.get(key, 0.0) + loss * n
            counts[key] = counts.get(key, 0) + n
    model.use_grad_checkpoint = grad_checkpoint
    model.train()
    return {f"val_loss_{k}": sums[k] / counts[k] for k in sums}


def main(args):
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out, "train.log"))],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rooms = load_rooms(args.rooms_train, args.limit_rooms)
    val_rooms = load_rooms(args.rooms_val, args.limit_rooms) if args.rooms_val else []
    assert max(r.num_targets + r.num_queries for r in rooms + val_rooms) <= args.max_group_size
    cache = FeatureCache(args.cache_dir, device)
    logger.info("rooms: %d (targets %d), validation rooms: %d", len(rooms), sum(r.num_targets for r in rooms), len(val_rooms))

    torch.manual_seed(args.seed)  # the layers that the loaded weights do not cover are initialised from the seed
    model = build_model(args).to(device)
    diffusion = create_diffusion("", noise_schedule="linear")
    autocast = make_autocast(args)
    optimizer = create_optimizer(model, args)

    latest = os.path.join(args.out, "latest.pt")
    hub = HubSync(args.hf_repo, args.hf_branch or os.path.basename(os.path.normpath(args.out))) if args.hf_repo else None
    if args.resume and not os.path.isfile(latest) and hub is not None:
        hub.download("latest.pt", latest)
    epoch, batch_index, global_step = 0, 0, 0
    if args.resume and os.path.isfile(latest):
        state = torch.load(latest, map_location="cpu")
        model.load_state_dict(state["model"])
        ema = SimpleEMAModel(model, decay=args.ema_decay)
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, batch_index, global_step = state["epoch"], state["batch_index"], state["global_step"]
        logger.info("resumed from %s at epoch %d, batch %d, step %d", latest, epoch, batch_index, global_step)
    else:
        if args.init_ckpt:
            load_pretrained_dit(model, args.init_ckpt)
        zero_sample_embedding(model)
        ema = SimpleEMAModel(model, decay=args.ema_decay)
    with open(os.path.join(args.out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=1)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("form %s, query_members %d, query_embedding %d, parameters %.1fM", args.form, args.query_members,
                args.query_embedding, num_params / 1e6)

    model.train()
    last_save = time.time()
    window_loss, window_targets, window_images, window_start = 0.0, 0, 0, time.time()
    done = False
    while epoch < args.epochs and not done:
        plan = plan_epoch(rooms, args.form, args.batch_size, args.seed, epoch, args.group_ratio)
        while batch_index < len(plan):
            batch = plan[batch_index]
            optimizer.zero_grad(set_to_none=True)
            loss, n, images = batch_loss(
                model, diffusion, cache, rooms, batch, args, step_seed(args.seed, global_step), autocast, True
            )
            if not math.isfinite(loss):
                raise RuntimeError(f"loss is {loss} at step {global_step}")
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.step(model)
            batch_index += 1
            global_step += 1
            window_loss += loss * n
            window_targets += n
            window_images += images

            if global_step % args.log_every == 0:
                elapsed = time.time() - window_start
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": window_loss / window_targets,
                    "targets_per_sec": window_targets / elapsed,
                    "images_per_sec": window_images / elapsed,
                }
                if device.type == "cuda":
                    record["max_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
                logger.info(json.dumps(record))
                with open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                window_loss, window_targets, window_images, window_start = 0.0, 0, 0, time.time()

            if time.time() - last_save > args.save_minutes * 60:
                save_checkpoint(latest, model, ema, optimizer, epoch, batch_index, global_step, args)
                if hub is not None:
                    hub.upload(latest, "latest.pt")
                last_save = time.time()
            if args.max_steps and global_step >= args.max_steps:
                done = True
                break
        if done:
            break
        epoch, batch_index = epoch + 1, 0
        if val_rooms and (epoch % args.val_every == 0 or epoch == args.epochs):
            record = {"epoch": epoch, "step": global_step, **validate(model, diffusion, cache, val_rooms, args, autocast)}
            logger.info(json.dumps(record))
            with open(os.path.join(args.out, "val_loss.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    save_checkpoint(latest, model, ema, optimizer, epoch, batch_index, global_step, args)
    final = os.path.join(args.out, "final_ema.pt")
    if epoch >= args.epochs:
        # what stays after training: the EMA weights and the arguments needed to rebuild the model
        torch.save({"ema": ema.state_dict(), "args": vars(args), "global_step": global_step}, final)
    if hub is not None:
        hub.upload(latest, "latest.pt", wait=True)
        for name in ["final_ema.pt", "log.jsonl", "val_loss.jsonl", "args.json", "train.log"]:
            path = os.path.join(args.out, name)
            if os.path.isfile(path):
                hub.upload(path, name, wait=True)
    logger.info("finished at epoch %d, step %d", epoch, global_step)


def get_args_parser():
    parser = argparse.ArgumentParser("Group training of a text-conditional DiT on rooms")
    # what is trained
    parser.add_argument("--form", required=True, choices=["none", "l", "f"], help="no groups, GroupDiff-l or GroupDiff-f")
    parser.add_argument("--query_members", default=0, type=int, choices=[0, 1], help="clean query images join the group")
    parser.add_argument("--query_member_drop", default=0.5, type=float, help="probability of a group without them")
    parser.add_argument("--query_embedding", default=0, type=int, choices=[0, 1], help="query set as image tokens")
    parser.add_argument("--cond_drop", default=0.1, type=float, help="probability of dropping text and image tokens")
    parser.add_argument("--group_ratio", default=0.1, type=float, help="share of the data in group batches (form l)")
    parser.add_argument("--random_flip", default=0, type=int, choices=[0, 1], help="needs a cache computed with --flip")

    # data
    parser.add_argument("--rooms_train", required=True, type=str)
    parser.add_argument("--rooms_val", default="", type=str)
    parser.add_argument("--cache_dir", required=True, type=str)
    parser.add_argument("--limit_rooms", default=0, type=int, help="use the first rooms only (for tests)")

    # model
    parser.add_argument("--model_size", default="xl", type=str, choices=list(SIZE_DICT))
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--max_group_size", default=20, type=int, help="rows of the sample embedding")
    parser.add_argument("--init_ckpt", default="", type=str, help="class-conditional DiT weights to start from")
    parser.add_argument("--grad_checkpointing", default=1, type=int, choices=[0, 1])
    parser.add_argument("--precision", default="bf16", type=str, choices=["bf16", "fp32"])

    # optimization
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help="targets per optimizer step")
    parser.add_argument("--chunk_images", default=0, type=int, help="images per forward pass; 0 for no limit")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--grad_clip", default=3.0, type=float)
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_steps", default=0, type=int, help="stop early (for tests)")

    # output
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--resume", action="store_true", help="continue from latest.pt of --out or of the hub branch")
    parser.add_argument("--val_every", default=10, type=int, help="epochs between validation losses")
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--save_minutes", default=20.0, type=float)
    parser.add_argument("--hf_repo", default="", type=str, help="private Hugging Face repository for checkpoints")
    parser.add_argument("--hf_branch", default="", type=str, help="branch of the run; defaults to the name of --out")
    return parser


if __name__ == "__main__":
    main(get_args_parser().parse_args())
