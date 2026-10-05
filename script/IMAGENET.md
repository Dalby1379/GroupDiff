# Furniture with the released class-conditional weights

The released GroupDiff weights take an ImageNet-1k class label as their only condition. This directory expresses
the furniture categories of the DeepFurniture retrieval benchmark as ImageNet-1k classes and generates images
with one class per image, so that the images of a group can be the different items of a room. Nothing is trained.

## Files

| File | Role |
|---|---|
| `imagenet_labels.py` | Runs ImageNet-1k classifiers on the item images, reports the classes of every furniture category, and writes group files. |
| `generate_imagenet.py` | Generation with the released weights from a group file, one class per image. |
| `check_imagenet.py` | Checks on a small random model that run on a CPU in a few seconds. |

## Usage

```bash
# 1. classes of the item images (two classifiers by default), then the classes of every category
python -m script.imagenet_labels classify --image_dir furnitures --metadata furnitures.jsonl \
    --categories categories.json --out labels
python -m script.imagenet_labels report --out labels --rooms rooms_train.json --tag train

# 2. group files: the targets of every room with the class of their category, or groups of one class
python -m script.imagenet_labels groups --classes labels/classes_train_<classifier>.json \
    --rooms rooms_test.json --out groups_rooms.json
python -m script.imagenet_labels groups --classes labels/classes_train_<classifier>.json --out groups_classes.json

# 3. generation
python -m script.generate_imagenet --ckpt released_model/gdiff-l-4-dit-xl-2-resume.pth \
    --groups groups_rooms.json --cfg 1.65 --out work_dirs/imagenet_rooms

# checks
python -m script.check_imagenet
```

`classes_<tag>_<classifier>.json` holds, for every category, the class that is the first class of the most items.
It is a plain `{category: class}` file; edit it to use other classes.

## Notes

- The released weights are GroupDiff-l. The conditional prediction is made for each image alone; the images of a
  group attend to each other only in the unconditional prediction. A class therefore steers its own image, and
  reaches the other images of the group only through the latent of its image.
- With `--cfg 1.0` there is no unconditional prediction, so the images of a group do not interact.
  `--group_attention 0` keeps the guidance and makes the unconditional prediction for each image alone.
- A group holds at most 4 images, the rows of the sample embedding of the released weights. The `groups` step
  leaves out rooms with more targets and logs how many; `generate_imagenet.py` refuses larger groups.
- The noise of an image is drawn from `--seed` + its position in the group file, so an image starts from the same
  noise with and without group attention.
- The released code draws the ids of the sample embedding at random in every call of the denoiser. With
  `--sample_ids random` (the default) an image therefore also depends on the other images of its batch, and
  `--group_attention 1` and `0` differ in these ids as well as in the attention. A rerun still gives the same
  images, because the global generator is pinned per batch. `--sample_ids position` gives an image the id of its
  position in its group in every call; an image then depends only on its group.
- Only the DiT weights are supported. The SiT weights use a different sampler (`models/sit.py`).
