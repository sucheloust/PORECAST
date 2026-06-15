# PORECAST — Training

nuclear **PORE** **C**omplex **A**utomated **S**egmentation **T**ool

3D U-Net for nuclear pore complex (NPC) segmentation in pan-ExM.

This is the original single training script split into focused modules. The
modules import each other by name, so run commands from inside this `training/`
directory (it is placed on `sys.path` automatically when you run `train.py`).

## Modules

| File | Contents |
|------|----------|
| `schedules.py` | Constant / linear-ramp hyperparameter schedules and the CLI parser. |
| `augment.py` | 3D augmentations: cube/SO(3) rotations, affine rescale, B-spline elastic deformation, random bias fields. |
| `preprocessing.py` | Per-volume normalization stats, Immerkaer noise estimation, CLAHE weak-signal filter. |
| `losses.py` | Soft clDice, focal-Tversky + SCE + GCE compound loss, focal+Dice fallback. |
| `model.py` | U-Net blocks (`DoubleConv3D`, `AASPP3D`), `UNet3D`, and the `EMAModel` wrapper. |
| `data.py` | `NPCDataset3D`, patch extraction, paired-folder partitioning, local-disk staging, stratified sampler. |
| `evaluate.py` | Patch + full-volume metrics, sliding-window inference, flip TTA, per-NPC detection F1. |
| `train.py` | Training loop, checkpointing, and the command-line entry point. |

## Install

Dependencies are pinned at the repository root:

```bash
pip install -r ../requirements.txt
```

## Train

From a single flat folder of paired image/mask TIFFs:

```bash
python train.py \
    --paired-dir /path/to/volumes \
    --paired-image-suffix pan_processed \
    --paired-mask-suffix antibody_mask \
    --out unet3d_npc.pt
```

Or from a pre-partitioned layout (`train/{images,masks}`, `val/{images,masks}`):

```bash
python train.py --train_dir /path/to/train --val_dir /path/to/val
```

Run `python train.py --help` for the full list of options (model size, loss
schedules, augmentation toggles, noise conditioning, staging, and validation
settings).
