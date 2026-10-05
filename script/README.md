# Group training on rooms of furniture items

This directory trains a text-conditional DiT on rooms of the DeepFurniture retrieval benchmark. A room is one scene,
split into query items and target items. The targets of a room are generated together as one group, each with its
own category prompt (`a <category>`), and the query items condition them.

`models/denoiser.py` gains an optional cross-attention layer in every block, between self-attention and the MLP.
It is off by default (`context_dim=0`), so the class-conditional models and the released weights behave as before.

## Files

| File | Role |
|---|---|
| `data.py` | Room files, the plan of an epoch, the feature cache. Run as a module, it computes the features. |
| `train.py` | Training with the three forms below, validation loss, checkpoints that resume exactly. |
| `generate.py` | Generation of the targets of every room with classifier-free guidance. |
| `check.py` | Checks on synthetic data that run on a CPU in a few seconds. |

## Forms

| `--form` | Conditional prediction | Unconditional prediction | Training batches |
|---|---|---|---|
| `none` | each target alone | each target alone | all single; conditions dropped for 10% of the targets |
| `l` (GroupDiff-l) | each target alone | targets of a room together | 10% of the data in unconditional group batches, 90% in conditional single batches |
| `f` (GroupDiff-f) | targets of a room together | targets of a room together | all group; conditions dropped for 10% of the rooms |

The query items enter in up to two ways:

- `--query_embedding 1`: the mean pooled image embedding of the query items (OpenCLIP ViT-bigG/14, as IP-Adapter
  computes it) is projected to four tokens and appended to the text tokens.
- `--query_members 1`: the clean latents of the query items join the group at timestep 0. They carry no loss and
  are left out of a group with probability `--query_member_drop` (0.5), so that generation can run with or
  without them.

## Usage

Room files (`rooms_train.json`, `rooms_val.json`, `rooms_test.json`) come from the exporter of the retrieval code
base; the format is described at the top of `data.py`.

```bash
# 1. features: VAE latents (256 x 256), pooled image embeddings, text token features (CLIP ViT-L/14)
python -m script.data --image_dir furnitures --rooms rooms_train.json rooms_val.json rooms_test.json --out cache

# 2. training, starting from the class-conditional DiT-XL/2 of https://github.com/facebookresearch/DiT
python -m script.train --form f --query_embedding 1 \
    --rooms_train rooms_train.json --rooms_val rooms_val.json --cache_dir cache \
    --init_ckpt DiT-XL-2-256x256.pt --out work_dirs/embed_f --resume

# 3. generation of the targets of the test rooms
python -m script.generate --ckpt work_dirs/embed_f/final_ema.pt --rooms rooms_test.json --cache_dir cache \
    --cfg 3.5 --out work_dirs/embed_f/test_cfg3.5

# checks
python -m script.check
```

`--resume` continues from `latest.pt` in `--out`. With `--hf_repo`, the latest checkpoint is also kept on a branch
of a private Hugging Face repository and fetched from there when `--out` has none.

## Notes

- `--batch_size` counts targets, the images that carry a loss. Query members come on top.
- Rooms have between 4 and 20 items, and the group attention of the released code needs one group size per
  forward pass. A batch is therefore split by group shape, and the gradients of the parts are accumulated before
  the single optimizer step. `--chunk_images` bounds the images of one forward pass when memory is short.
- The output projection of the cross-attention and the sample embedding start at zero. With the class label fixed
  to the null class, the model starts with exactly the outputs of the weights it was loaded from.
- The targets of a group share one timestep. Every optimizer step draws its noise, timesteps and dropped
  conditions from the seed and the step number, which is what makes a resumed run identical to an uninterrupted one.
- Latents are one posterior sample per image scaled by 0.18215, the convention of `dataset/extract_feats.py`.
  Mirrored latents are stored only with `--flip` and used only with `--random_flip 1`.
- In generation, the noise of a target is drawn from `--seed + sample_index`, so a target starts from the same
  noise whether it is sampled alone or in its group.
