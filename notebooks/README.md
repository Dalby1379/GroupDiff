# Colab notebook for the DiT runs on rooms of furniture items

`heterogeneous_diffusion.ipynb` prepares a Colab VM, trains one run of `script/train.py`, chooses the guidance weight,
generates test split 0 and scores it. Everything it runs is in this repository at the commit written in its first
cell; the data it reads are on a private Hugging Face repository, checked against `hf_files.sha256`.

The scripts are the ones that produced the recorded runs (Colab scripts of the retrieval code base, SetRetrieval
49e30bd `methods/heterogeneous_diffusion/colab/`), with the same commands and the same paths on the VM. What changed:
the code no longer comes from another repository, the room files and the test gallery come from a data branch instead
of an upload, and the final ranks are computed on the VM (`script/evaluate.py`, clip only) instead of on a server.

## The four experiments of the results table

| Experiment | `RUN` | Weight (`CFG`) | Test output |
|---|---|---|---|
| (a) image embedding, one target at a time | `dit_embed_none` | 2.5 | `test0_members0` |
| (a) image embedding, GroupDiff-f | `dit_embed_f` | 5.0 | `test0_members0` |
| (b) query images in the group, generated without them, GroupDiff-f | `dit_members_f` | 5.0 | `test0_members0` |
| (b) query images in the group, generated with them, GroupDiff-f | `dit_members_f` | 5.0 | `test0_members1` |

`CFG = None` chooses the weight again on the 100 validation rooms; with GPU generation that is a new measurement and
can pick another weight (two of the recorded choices were ties).

## Files

| File | Role |
|---|---|
| `heterogeneous_diffusion.ipynb` | The notebook. One cell is one blocking command; cells can be run again |
| `hd_common.py` | Run names, paths on the VM, sha256 checks, downloads from the private repository |
| `setup_vm.py` | Token, dependencies, `script/check.py` on the CPU, data, feature cache, DiT-XL/2 weights |
| `train_run.py` | Training of one run (`--smoke` for 20 updates); resumes from the latest checkpoint |
| `eval_run.py` | Choice of the guidance weight, generation of test split 0, clip embeddings, ranks |
| `hf_files.sha256` | sha256 of every downloaded file |
| `requirements-colab.txt` | Versions of the dependencies (torch and Python come with the Colab image) |
| `verify_port.py` | CPU checks against the recorded runs (ranks, weight choice, commands, room loader) |
| `tools/hf_prune_checkpoints.py` | Removes replaced checkpoints from the private repository; runs outside the notebook |

## Before running

- Colab secret `HF_TOKEN` with access to the private repository.
- `CODE_COMMIT` in the first cell is the full 40-character hash of the code commit. The notebook cannot name its own commit, so the code is
  committed first and the notebook pinned to it in a second commit; any change of the code needs a new pin.
- Training uploads a 13 GB `latest.pt` every 20 minutes. Run `tools/hf_prune_checkpoints.py --execute --loop_minutes 5`
  on another machine for the whole training, or the 100 GB quota of the private repository fills within hours.
- A branch `run-<RUN>` that already holds `final_ema.pt` is not trained again (the recorded branches no longer hold
  `latest.pt`, so `--resume` would start from scratch and replace the recorded weights). Use `--hf_branch` for a new run.
  `eval_run.py --upload` likewise never writes to an existing `eval-<RUN>` branch.
- Two gallery files have `#` in their names (`Cabinet#Shelf.pkl`, `Chair#Stool.pkl`). If a download by that name fails,
  the files can be renamed on the data branch without changing any result: `script/evaluate.py` takes the category from
  `category_name` inside the pkl, not from the file name (the names in `hd_common.py` and `hf_files.sha256` then change too).
