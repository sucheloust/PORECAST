#!/usr/bin/env python3
"""
NPC 3D U-Net training (v9).

Inherits all v8 machinery (Focal-Tversky + SCE + GCE + clDice, confident-
learning label cleaning, noise FiLM conditioning, SO(3)/elastic augmentation,
AASPP bottleneck, CLAHE path). Six targeted fixes for the 1-3 ring-per-NPC
ambiguity and several code correctness issues:

  (1) AASPP bottleneck FiLM bug fixed. v8 accepted the noise-conditioning
      vector in AASPP3D.forward but silently dropped it; FiLM scale/shift is
      now applied after the fuse step, matching DoubleConv3D behaviour.

  (2) Distance-heatmap auxiliary supervision (--w-dist, default 0.1). A
      Gaussian-blurred version of the binary mask (sigma --dist-sigma,
      default 4 voxels) is used as the regression target for the existing
      dist output head, which was previously unsupervised dead weight. The
      heatmap is topology-agnostic: a face-on disk, 2-ring, and 3-ring NPC
      all produce a similar centroid-centred response, so the head provides
      a consistent shape prior regardless of which view the patch samples.
      MSE(sigmoid(dist_pred), gaussian_target) added with weight w_dist.

  (3) NPC-level detection metric (--npc-group-radius, default 20 voxels).
      v8 detection F1 ran connected-component IoU at the ring level, so a
      3-ring NPC counted as 3 separate GT objects. Any prediction covering
      the correct location but with different topology (blob vs 3 rings)
      incurred spurious FP/FN. v9 dilates both pred and GT binary masks by
      npc_group_radius before labelling, merging co-NPC rings into one
      instance on both sides before IoU matching. det_f1 now measures NPC
      localisation, not ring topology.

  (4) Deep-supervision heads use pos_weight. v8 DS heads used plain BCE,
      which is dominated by the vast background at fg_fraction ~0.0015. v9
      passes criterion.pos_weight to DS BCE, matching the main loss signal.

  (5) LR warmup (--lr-warmup-epochs, default 5). LinearLR warmup then
      CosineAnnealingLR via SequentialLR. Stabilises early training with
      the 6-term compound loss.

  (6) clean-max-frac default 0.10 -> 0.20; full-val-roi default 256 -> 512.

Sliding-window full-volume validation (--full-val-every) reports stitched
Dice/IoU and per-NPC detection F1 (NPC-level). Built for large volumes
(~400 x 2720 x 2720).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import hashlib
import pickle
from pickle import PicklingError
import multiprocessing as mp
from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.ndimage import (label as label_3d,
                           white_tophat, uniform_filter, convolve,
                           gaussian_filter, binary_dilation)
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# Hyperparameter schedules: constant, or linear ramp "start:end:epochs".
# Epoch is the training epoch (continues across --resume).
class ConstantSchedule:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, epoch):
        return self.value

    def __repr__(self):
        return f"const({self.value})"


class LinearSchedule:
    def __init__(self, start, end, epochs):
        self.start = float(start)
        self.end = float(end)
        self.epochs = int(epochs)

    def __call__(self, epoch):
        if self.epochs <= 0 or epoch >= self.epochs:
            return self.end
        if epoch <= 0:
            return self.start
        frac = epoch / self.epochs
        return self.start + (self.end - self.start) * frac

    def __repr__(self):
        return f"linear({self.start}->{self.end} over {self.epochs}ep)"


def parse_schedule(spec):
    """
    Parse a CLI schedule spec into a schedule object.
      "0.5"          -> ConstantSchedule(0.5)
      "0.3:0.5:40"   -> LinearSchedule(0.3, 0.5, 40)
    """
    spec = str(spec).strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Schedule '{spec}' must be 'start:end:epochs' or a single "
                f"number.")
        start, end, epochs = parts
        return LinearSchedule(float(start), float(end), int(epochs))
    return ConstantSchedule(float(spec))




def _detect_paired_layout(folder, image_marker, mask_marker):
    """
    Find paired (image, mask) files in a flat folder. For each file
    containing image_marker, look for a sibling with image_marker
    replaced by mask_marker. Requires every image-side file to have a
    mask-side partner (and vice versa).
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Paired folder not found: {folder}")
    all_tifs = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    if not all_tifs:
        raise FileNotFoundError(f"No .tif/.tiff files in {folder}")

    names = [f.name for f in all_tifs]
    by_name = {f.name: f for f in all_tifs}

    pairs = []
    for name in names:
        if image_marker in name:
            mask_name = name.replace(image_marker, mask_marker)
            if mask_name in by_name and mask_name != name:
                pairs.append((by_name[name], by_name[mask_name]))
    n_img = sum(1 for n in names if image_marker in n)
    n_mask = sum(1 for n in names if mask_marker in n)
    if not (len(pairs) >= 2 and len(pairs) == n_img == n_mask):
        sample = "\n  ".join(names[:8])
        raise ValueError(
            f"Pattern '{image_marker}' / '{mask_marker}' did not cleanly "
            f"pair files in {folder}.\n"
            f"  {n_img} files contain '{image_marker}', "
            f"{n_mask} files contain '{mask_marker}', "
            f"{len(pairs)} valid pairs after substitution.\n"
            f"  First few files:\n  {sample}")
    pairs.sort(key=lambda p: p[0].name)
    print(f"[paired-detect] Found {len(pairs)} pairs using "
          f"'{image_marker}' -> '{mask_marker}' in {folder}")
    return [p[0] for p in pairs], [p[1] for p in pairs]


def partition_train_val_paired(paired_dir, output_root,
                               val_fraction=0.15, seed=42,
                               image_marker=None, mask_marker=None):
    """
    Like partition_train_val but for a single flat folder with paired
    files (e.g. vol_01_img.tif + vol_01_mask.tif). Creates the same
    train/{images,masks} + val/{images,masks} structure so the rest of
    the pipeline is unchanged.

    If image_marker and mask_marker are both provided, the auto-detect
    is bypassed and that pattern is used directly.
    """
    paired_dir = Path(paired_dir)
    output_root = Path(output_root)
    train_dir, val_dir = output_root / "train", output_root / "val"

    # Skip if already partitioned
    if (train_dir / "images").exists() and (val_dir / "images").exists():
        n_tr = len(list((train_dir / "images").glob("*.tif*")))
        n_va = len(list((val_dir / "images").glob("*.tif*")))
        if n_tr > 0 and n_va > 0:
            print(f"[partition] Split exists: {n_tr} train, {n_va} val. Skipping.")
            return train_dir, val_dir

    img_files, mask_files = _detect_paired_layout(
        paired_dir, image_marker=image_marker, mask_marker=mask_marker)
    assert len(img_files) >= 2, f"Need >=2 pairs, found {len(img_files)}."

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(img_files))
    n_val = max(1, int(len(img_files) * val_fraction))
    val_set = set(idx[:n_val])

    for d in [train_dir, val_dir]:
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "masks").mkdir(parents=True, exist_ok=True)

    # Symlink into the conventional layout. We preserve the original
    # filenames (so vol_01_img.tif goes to train/images/vol_01_img.tif).
    # The dataset matches images to masks by sorted order within each
    # subdir, which works as long as paired files sort consistently --
    # which they do by construction since we detected them via paired
    # marker substitution.
    for i, (ip, mp) in enumerate(zip(img_files, mask_files)):
        dest = val_dir if i in val_set else train_dir
        for sub, src in [("images", ip), ("masks", mp)]:
            dst = dest / sub / src.name
            if dst.exists():
                continue
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(str(src), str(dst))

    print(f"[partition] Created: {len(img_files)-n_val} train, "
          f"{n_val} val in {output_root}")
    return train_dir, val_dir


# ---------------------------------------------------------------------------
# GPU Keepalive
# ---------------------------------------------------------------------------
def start_gpu_keepalive():
    script = ('import torch,time,sys\n'
              'if not torch.cuda.is_available(): sys.exit(0)\n'
              'd=torch.device("cuda")\n'
              'while True:\n'
              ' try: a=torch.randn(256,256,device=d);torch.mm(a,a);'
              'torch.cuda.synchronize();del a\n'
              ' except: pass\n'
              ' time.sleep(30)\n')
    return subprocess.Popen([sys.executable, '-c', script],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_gpu_keepalive(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# 24 proper rotations of the cube
# ---------------------------------------------------------------------------
CUBE_ROTATIONS = [
    [[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]], [[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]],
    [[-1, 0, 0],[ 0, 1, 0],[ 0, 0,-1]], [[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]],
    [[ 1, 0, 0],[ 0, 0, 1],[ 0,-1, 0]], [[ 1, 0, 0],[ 0, 0,-1],[ 0, 1, 0]],
    [[-1, 0, 0],[ 0, 0, 1],[ 0, 1, 0]], [[-1, 0, 0],[ 0, 0,-1],[ 0,-1, 0]],
    [[ 0, 1, 0],[ 1, 0, 0],[ 0, 0,-1]], [[ 0, 1, 0],[-1, 0, 0],[ 0, 0, 1]],
    [[ 0,-1, 0],[ 1, 0, 0],[ 0, 0, 1]], [[ 0,-1, 0],[-1, 0, 0],[ 0, 0,-1]],
    [[ 0, 1, 0],[ 0, 0, 1],[ 1, 0, 0]], [[ 0, 1, 0],[ 0, 0,-1],[-1, 0, 0]],
    [[ 0,-1, 0],[ 0, 0, 1],[-1, 0, 0]], [[ 0,-1, 0],[ 0, 0,-1],[ 1, 0, 0]],
    [[ 0, 0, 1],[ 1, 0, 0],[ 0, 1, 0]], [[ 0, 0, 1],[-1, 0, 0],[ 0,-1, 0]],
    [[ 0, 0,-1],[ 1, 0, 0],[ 0,-1, 0]], [[ 0, 0,-1],[-1, 0, 0],[ 0, 1, 0]],
    [[ 0, 0, 1],[ 0, 1, 0],[-1, 0, 0]], [[ 0, 0, 1],[ 0,-1, 0],[ 1, 0, 0]],
    [[ 0, 0,-1],[ 0, 1, 0],[ 1, 0, 0]], [[ 0, 0,-1],[ 0,-1, 0],[-1, 0, 0]],
]


def apply_cube_rotation(vol, R):
    perm, flips = [], []
    for i in range(3):
        j = int(np.argmax(np.abs(R[i])))
        perm.append(j)
        if R[i][j] < 0:
            flips.append(i)
    out = np.transpose(vol, perm)
    for ax in flips:
        out = np.flip(out, axis=ax)
    return out.copy()


# Arbitrary-angle SO(3) rotation + isotropic rescale. Cube rotations only
# permute axes; SO(3) exposes intermediate orientations (e.g. top-down NPCs).
# Rescale resamples patches in [s_min, s_max] for cross-expansion-factor
# robustness.
def _random_rotation_matrix_so3(rng=None):
    """Uniformly-random rotation matrix on SO(3) via Shoemake's quaternion method."""
    if rng is None:
        rng = np.random
    u1, u2, u3 = rng.uniform(0, 1, 3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1)     * np.sin(2 * np.pi * u3),
        np.sqrt(u1)     * np.cos(2 * np.pi * u3),
    ], dtype=np.float64)
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def apply_affine_3d(img, mask, R=None, scale=1.0, scale_z=None):
    """
    Apply a combined rotation + anisotropic-scale affine transform to a 3D
    image and its mask using torch.nn.functional.grid_sample.

    img, mask: numpy arrays of shape (D, H, W), float32.
    R: 3x3 rotation matrix (numpy). If None, identity.
    scale: XY isotropic scale factor (>1 = zoom in).
    scale_z: independent Z scale factor; if None, falls back to scale.
    Returns (img_out, mask_out) as numpy float32 arrays of the same shape.

    The image uses bilinear interpolation; the mask uses nearest-neighbor
    so labels are not blurred. Out-of-bounds samples use reflection padding
    to avoid black borders that would bias the network.
    """
    if R is None:
        R = np.eye(3, dtype=np.float32)
    D, H, W = img.shape

    # grid_sample uses normalized coords in [-1, 1] with axis order (W, H, D)
    # for a (N, C, D, H, W) input. The affine theta therefore operates in
    # (x, y, z) = (W, H, D) ordering. We want the *inverse* map
    # (output -> input), so use R^T; scale > 1 means sampling more tightly
    # (zoom in). Anisotropic Z/XY scaling teaches disk orientation invariance:
    # compressing Z (scale_z > scale) simulates face-on ring morphology;
    # stretching Z (scale_z < scale) simulates side-on 3-dot morphology.
    sz = float(scale_z) if scale_z is not None else float(scale)
    sxy = float(scale)
    # S_inv in (D, H, W) order then composed with R^T for the inverse map.
    s_inv = np.array([1.0 / sz, 1.0 / sxy, 1.0 / sxy], dtype=np.float32)
    R_inv = s_inv[:, None] * R.T.astype(np.float32)   # S_inv @ R^T
    theta = np.zeros((1, 3, 4), dtype=np.float32)
    # Reorder axes from (D, H, W) -> (W, H, D): swap rows 0 and 2, and
    # cols 0 and 2, of R_inv so theta operates in grid_sample's coords.
    perm = np.array([2, 1, 0])
    theta[0, :3, :3] = R_inv[perm][:, perm]

    img_t   = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    mask_t  = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    theta_t = torch.from_numpy(theta)

    grid = F.affine_grid(theta_t, img_t.shape, align_corners=False)
    img_out = F.grid_sample(img_t, grid, mode='bilinear',
                            padding_mode='reflection', align_corners=False)
    mask_out = F.grid_sample(mask_t, grid, mode='nearest',
                             padding_mode='reflection', align_corners=False)
    return (img_out.squeeze().numpy().astype(np.float32),
            mask_out.squeeze().numpy().astype(np.float32))


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def _soft_erode3d(x):
    p1 = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_open3d(x):
    return F.max_pool3d(_soft_erode3d(x), 3, 1, 1)


def soft_skeletonize3d(x, iters=3):
    """Differentiable morphological skeleton (Shit et al. 2021, clDice)."""
    skel = F.relu(x - _soft_open3d(x))
    for _ in range(iters):
        x = _soft_erode3d(x)
        delta = F.relu(x - _soft_open3d(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(p, target_mask, iters=3, smooth=1.0):
    """Soft centerline-Dice; rewards ring/tubular connectivity."""
    sk_p = soft_skeletonize3d(p, iters)
    sk_t = soft_skeletonize3d(target_mask, iters)
    tprec = (sk_p * target_mask).sum() + smooth
    tprec = tprec / (sk_p.sum() + smooth)
    tsens = (sk_t * p).sum() + smooth
    tsens = tsens / (sk_t.sum() + smooth)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


class TverskyDiceLoss(nn.Module):
    """
    Focal-Tversky + symmetric-cross-entropy + soft-clDice + distance MSE,
    with scheduled hyperparameters. alpha/beta/w_bce/w_cldice/pos_weight may
    each be a schedule object (ConstantSchedule or LinearSchedule);
    set_epoch(ep) updates the live values at the start of each epoch.

    The cross-entropy term is SCE (Wang et al. 2019): w_bce weights the
    sum of pos-weighted CE and reverse-CE (RCE), the latter bounded and
    robust to symmetric mask label noise. Tversky is made focal
    (Abraham & Khan 2018) via (1-TI)**ft_gamma to emphasise hard patches.
    clDice (warmed up via its schedule) adds ring-topology supervision.

    w_gce adds a GCE/Lq robust term (Zhang & Sabuncu 2018) on foreground
    voxels: L_q = (1 - p^q)/q, which interpolates between CE (q->0) and the
    noise-robust MAE (q=1). Its gradient is bounded, so spurious positive
    labels (antibody false positives, the v8 noise regime) cannot dominate
    the loss the way an aggressive pos-weighted BCE term does.
    """
    def __init__(self, w_bce_sched, w_tversky=1.0,
                 pos_weight_sched=None, alpha_sched=None, beta_sched=None,
                 w_cldice_sched=None, smooth=1.0, ft_gamma=1.333,
                 sce_alpha=1.0, sce_beta=0.5, rce_clip=-4.0, cldice_iters=3,
                 w_gce=0.0, gce_q=0.7):
        super().__init__()
        self.w_tversky = w_tversky
        self.smooth = smooth
        self.ft_gamma = float(ft_gamma)
        self.sce_alpha, self.sce_beta = float(sce_alpha), float(sce_beta)
        self.rce_clip = float(rce_clip)
        self.cldice_iters = int(cldice_iters)
        self.w_gce = float(w_gce)
        self.gce_q = float(gce_q)
        # Schedules
        self._w_bce_sched = w_bce_sched
        self._pos_weight_sched = pos_weight_sched
        self._alpha_sched = alpha_sched
        self._beta_sched = beta_sched
        self._w_cldice_sched = w_cldice_sched or ConstantSchedule(0.0)
        # Live values, initialized to epoch 0
        self.w_bce = w_bce_sched(0)
        self.alpha = alpha_sched(0)
        self.beta = beta_sched(0)
        self.w_cldice = self._w_cldice_sched(0)
        self.register_buffer('pos_weight',
                             torch.tensor([pos_weight_sched(0)]))

    def set_epoch(self, epoch):
        """Update scheduled hyperparameters for the given training epoch."""
        self.w_bce = self._w_bce_sched(epoch)
        self.alpha = self._alpha_sched(epoch)
        self.beta = self._beta_sched(epoch)
        self.w_cldice = self._w_cldice_sched(epoch)
        self.pos_weight.fill_(self._pos_weight_sched(epoch))

    def current_values(self):
        return {'w_bce': self.w_bce, 'alpha': self.alpha,
                'beta': self.beta, 'w_cldice': self.w_cldice,
                'pos_weight': float(self.pos_weight.item())}

    def forward(self, pred_mask, target_mask, weight_map=None):
        p = torch.sigmoid(pred_mask)
        # SCE = sce_alpha * CE + sce_beta * RCE. For binary targets with
        # log(0) clamped to rce_clip, RCE reduces to |rce_clip|*(p+t-2pt).
        ce_raw = F.binary_cross_entropy_with_logits(
            pred_mask, target_mask, pos_weight=self.pos_weight,
            reduction='none').clamp(max=100.0)
        rce_raw = abs(self.rce_clip) * (p + target_mask - 2 * p * target_mask)
        sce_raw = self.sce_alpha * ce_raw + self.sce_beta * rce_raw
        if weight_map is not None:
            sce = (sce_raw * weight_map).mean()
        else:
            sce = sce_raw.mean()
        tp = (p * target_mask).sum()
        fp = (p * (1 - target_mask)).sum()
        fn = ((1 - p) * target_mask).sum()
        ti = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        # ti is mathematically in (0, 1], but under AMP/fp16 rounding it can
        # land a hair above 1 when fp and fn are both ~0 (near-perfect or
        # near-empty patch). (1 - ti) is then a tiny negative, and a negative
        # base raised to a fractional power (ft_gamma=1.333) is NaN. Clamp the
        # base to [0, 1] before the power to keep the focal Tversky finite.
        tversky = (1 - ti).clamp(min=0.0, max=1.0) ** self.ft_gamma
        cldice = (soft_cldice(p, target_mask, self.cldice_iters)
                  if self.w_cldice > 0 else torch.zeros((), device=p.device))
        # GCE/Lq robust positive loss on foreground-labelled voxels. Computed
        # in fp32 (pow under AMP/fp16 is unstable) and averaged over fg voxels
        # only, so its gradient is bounded and a few spurious positive labels
        # cannot swamp the signal from genuine NPCs.
        if self.w_gce > 0:
            fg = (target_mask > 0.5).float()
            n_fg = fg.sum()
            if n_fg > 0:
                p_t = p.float().clamp(min=1e-6, max=1.0)
                gce = (((1.0 - p_t ** self.gce_q) / self.gce_q)
                       * fg).sum() / n_fg
            else:
                gce = torch.zeros((), device=p.device)
        else:
            gce = torch.zeros((), device=p.device)
        total = (self.w_bce * sce + self.w_tversky * tversky
                 + self.w_cldice * cldice + self.w_gce * gce)
        return total, {'sce': sce.item(), 'tversky': tversky.item(),
                       'cldice': float(cldice.item()),
                       'gce': float(gce.item())}



class FocalDiceLoss(nn.Module):
    """Focal + Dice for collapse recovery."""
    def __init__(self, w_focal=1.0, w_dice=1.0,
                 alpha=0.75, gamma=2.0, dice_smooth=1.0):
        super().__init__()
        self.w_focal, self.w_dice = w_focal, w_dice
        self.alpha, self.gamma, self.dice_smooth = alpha, gamma, dice_smooth

    def set_epoch(self, epoch):
        """No-op: FocalDiceLoss has no scheduled hyperparameters. Present so
        the training loop can call criterion.set_epoch(ep) uniformly after
        collapse recovery swaps in this loss."""
        pass

    def current_values(self):
        return {}

    def forward(self, pred_mask, target_mask, weight_map=None):
        p = torch.sigmoid(pred_mask)
        bce = F.binary_cross_entropy_with_logits(
            pred_mask, target_mask, reduction='none')
        pt = torch.where(target_mask == 1, p, 1 - p)
        at = torch.where(target_mask == 1, self.alpha, 1 - self.alpha)
        focal = (at * (1 - pt)**self.gamma * bce).mean()
        inter = (p * target_mask).sum()
        dice = 1 - (2*inter + self.dice_smooth) / (
            p.sum() + target_mask.sum() + self.dice_smooth)
        total = self.w_focal*focal + self.w_dice*dice
        return total, {'focal': focal.item(), 'dice': dice.item()}



class DoubleConv3D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, cond_dim=0):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size, padding=p, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size, padding=p, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.residual = (nn.Conv3d(in_c, out_c, 1, bias=False)
                         if in_c != out_c else nn.Identity())
        # FiLM: noise-conditioned per-channel scale/shift applied AFTER the
        # InstanceNorm-ed feature map (a constant input channel would be
        # erased by InstanceNorm, so conditioning must modulate post-norm).
        self.film = nn.Linear(cond_dim, 2 * out_c) if cond_dim else None

    def forward(self, x, cond=None):
        h = self.net(x)
        if self.film is not None and cond is not None:
            g, b = self.film(cond).chunk(2, dim=1)
            g = g.view(g.shape[0], -1, 1, 1, 1)
            b = b.view(b.shape[0], -1, 1, 1, 1)
            h = h * (1 + g) + b
        return h + self.residual(x)


class AASPP3D(nn.Module):
    """
    3D Atrous Spatial Pyramid Pooling bottleneck (adapted from MSA-UNet, 2021).
    Three parallel dilated-conv branches (rates 1, 2, 4) + global avg-pool
    branch, concatenated and fused with 1×1 conv + residual shortcut.
    Replaces the plain DoubleConv3D bottleneck to widen the receptive field
    without adding depth. Requires out_c divisible by 4.

    v9: FiLM conditioning (cond_dim) wired up. v8 accepted cond but silently
    dropped it; now applied after fuse, matching DoubleConv3D behaviour.
    """
    def __init__(self, in_c, out_c, cond_dim=0):
        super().__init__()
        mid = out_c // 4
        self.b1 = self._branch(in_c, mid, dilation=1)
        self.b2 = self._branch(in_c, mid, dilation=2)
        self.b4 = self._branch(in_c, mid, dilation=4)
        self.bglobal = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_c, mid, 1, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(mid * 4, out_c, 1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.residual = nn.Conv3d(in_c, out_c, 1, bias=False)
        self.film = nn.Linear(cond_dim, 2 * out_c) if cond_dim else None

    @staticmethod
    def _branch(in_c, out_c, dilation):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x, cond=None):
        g = F.interpolate(self.bglobal(x), size=x.shape[2:],
                          mode='trilinear', align_corners=False)
        out = self.fuse(torch.cat([self.b1(x), self.b2(x), self.b4(x), g], dim=1))
        if self.film is not None and cond is not None:
            gf, b = self.film(cond).chunk(2, dim=1)
            gf = gf.view(gf.shape[0], -1, 1, 1, 1)
            b = b.view(b.shape[0], -1, 1, 1, 1)
            out = out * (1 + gf) + b
        return out + self.residual(x)


class UNet3D(nn.Module):
    def __init__(self, in_c=1, base=16, depth=4, dropout=0.1,
                 deep_supervision=True, use_aaspp=False,
                 noise_conditioning=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.depth = depth
        # Noise-level conditioning: a scalar per patch -> shared embedding ->
        # per-block FiLM. Lets the network adapt to acquisition noise (clean
        # vs noisy imaging dates) instead of averaging over both regimes.
        self.noise_conditioning = noise_conditioning
        cond_dim = base if noise_conditioning else 0
        self.cond_mlp = (nn.Sequential(
            nn.Linear(1, base), nn.SiLU(), nn.Linear(base, cond_dim))
            if noise_conditioning else None)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(depth):
            ic = in_c if i == 0 else base * (2 ** (i - 1))
            oc = base * (2 ** i)
            k = 5 if i < 2 else 3
            self.encoders.append(DoubleConv3D(ic, oc, kernel_size=k,
                                              cond_dim=cond_dim))
            if i < depth - 1:
                # Strided conv instead of MaxPool (nnU-Net, NISNet3D)
                self.downsamples.append(
                    nn.Conv3d(oc, oc, 2, stride=2, bias=False))

        bn_in = base * (2**(depth-1))
        bn_out = base * (2**depth)
        self.bottleneck = (AASPP3D(bn_in, bn_out, cond_dim=cond_dim) if use_aaspp
                           else DoubleConv3D(bn_in, bn_out, cond_dim=cond_dim))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            ic = base * (2 ** (i + 1))
            oc = base * (2 ** i)
            self.upconvs.append(nn.ConvTranspose3d(ic, oc, 2, stride=2))
            k = 5 if i < 2 else 3
            self.decoders.append(DoubleConv3D(ic, oc, kernel_size=k,
                                              cond_dim=cond_dim))

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.out_mask = nn.Conv3d(base, 1, 1)
        self.out_dist = nn.Conv3d(base, 1, 1)

        # Deep supervision heads at each decoder level
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for i in range(depth - 1, -1, -1):
                oc = base * (2 ** i)
                self.ds_heads.append(nn.Conv3d(oc, 1, 1))

    def forward(self, x, noise=None):
        target_shape = x.shape[2:]
        cond = None
        if self.cond_mlp is not None and noise is not None:
            cond = self.cond_mlp(noise.float().view(-1, 1))
        enc = []
        for i, encoder in enumerate(self.encoders[:-1]):
            x = encoder(x, cond)
            enc.append(x)
            x = self.downsamples[i](x)
        x = self.encoders[-1](x, cond)
        enc.append(x)
        x = self.bottleneck(x, cond)

        ds_outputs = []
        for idx, (upconv, decoder, skip) in enumerate(
                zip(self.upconvs, self.decoders, reversed(enc))):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = self._match(x, skip.shape[2:])
            x = decoder(torch.cat([x, skip], 1), cond)
            if self.deep_supervision and self.training:
                ds_logits = self.ds_heads[idx](x)
                if ds_logits.shape[2:] != target_shape:
                    ds_logits = F.interpolate(
                        ds_logits, size=target_shape,
                        mode='trilinear', align_corners=False)
                ds_outputs.append(ds_logits)

        x = self.dropout(x)
        return self.out_mask(x), self.out_dist(x), ds_outputs

    @staticmethod
    def _match(x, target):
        slices, pads = [], []
        for c, t in zip(x.shape[2:], target):
            if c > t:
                s = (c - t) // 2
                slices.append(slice(s, s + t)); pads.extend([0, 0])
            elif c < t:
                slices.append(slice(None))
                d = t - c; pads.extend([d // 2, d - d // 2])
            else:
                slices.append(slice(None)); pads.extend([0, 0])
        x = x[:, :, slices[0], slices[1], slices[2]]
        if any(pads):
            x = F.pad(x, pads[::-1])
        return x


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1-self.decay) * p.data

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}


# B-spline elastic deformation (nnU-Net / Cicek 2016): sample displacements
# on a coarse control grid ~ N(0, sigma), upsample to a dense smooth field,
# apply to image (linear) and mask (nearest).
def elastic_deform_3d(img, mask, control_spacing=32, sigma=4.0, mode='bspline'):
    """Apply a smooth dense displacement field identically to img and mask.
    mode='bspline' (cubic, C^2, slow) or 'trilinear' (fast, C^0)."""
    from scipy.ndimage import map_coordinates, zoom
    shape = img.shape
    # Control grid: >=4 points/axis (cubic support), +1 margin for upsample.
    ctrl_shape = tuple(max(4, int(np.ceil(s / control_spacing)) + 1)
                       for s in shape)

    # Upsample three independent displacement fields (one per axis).
    disp_full = []
    for _ in range(3):
        disp_ctrl = np.random.randn(*ctrl_shape).astype(np.float32) * sigma
        if mode == 'bspline':
            zoom_factors = tuple(s / c for s, c in zip(shape, ctrl_shape))
            d = zoom(disp_ctrl, zoom_factors, order=3,
                     mode='reflect', prefilter=True)
        elif mode == 'trilinear':
            t = torch.from_numpy(disp_ctrl).float().unsqueeze(0).unsqueeze(0)
            d = F.interpolate(t, size=shape, mode='trilinear',
                              align_corners=True).squeeze().numpy()
        else:
            raise ValueError(f"Unknown elastic mode: {mode}")
        # Trim/pad if upsampling rounded shape slightly off
        if d.shape != shape:
            slc = tuple(slice(0, s) for s in shape)
            d = d[slc]
        disp_full.append(d)

    out_grids = np.meshgrid(*[np.arange(s, dtype=np.float32)
                              for s in shape], indexing='ij')
    sample_coords = [np.clip(out_grids[a] + disp_full[a], 0, shape[a] - 1)
                     for a in range(3)]

    img_out = map_coordinates(img, sample_coords, order=1,
                              mode='reflect').astype(np.float32)
    mask_out = map_coordinates(mask, sample_coords, order=0,
                               mode='reflect').astype(np.float32)
    return img_out, mask_out


def random_bias_field_3d(img, max_coeff=0.3, ctrl=4):
    """Smooth multiplicative intensity field (microscopy illumination /
    staining inhomogeneity). A low-order random control grid is upsampled
    and exponentiated so the field is positive and slowly varying ~1."""
    coarse = np.random.uniform(-max_coeff, max_coeff,
                               (ctrl, ctrl, ctrl)).astype(np.float32)
    t = torch.from_numpy(coarse)[None, None]
    field = F.interpolate(t, size=img.shape, mode='trilinear',
                          align_corners=True)[0, 0].numpy()
    return (img * np.exp(field)).astype(np.float32)


def _extract_volume_patches(args):
    """Per-volume patch extraction worker (module-level for pickling)."""
    (img_p, mask_p, patch_size, min_npc_size, augment, jitter_range,
     add_negative_samples, negative_ratio,
     max_patches_per_volume, seed) = args

    rng = np.random.RandomState(seed)
    stats = compute_volume_stats_streaming(img_p)

    # memmap keeps mask pages file-backed so they can be evicted under pressure;
    # falls back to a full read for compressed TIFs.
    try:
        mask_vol = tifffile.memmap(str(mask_p), mode='r')
    except Exception:
        mask_vol = tifffile.imread(str(mask_p))
    Z, H, W = mask_vol.shape
    total_voxels = Z * H * W

    binary = mask_vol > 0
    if not add_negative_samples:
        del mask_vol  # not needed again; release pages before label_3d peaks
    labeled, n_npc = label_3d(binary)
    del binary  # free 1-byte/voxel bool before regionprops accumulates

    if n_npc == 0:
        if add_negative_samples:
            del mask_vol
        return {
            'img_p': str(img_p), 'mask_p': str(mask_p),
            'stats': stats, 'patches': [], 'fg_flags': [],
            'n_components_total': 0, 'n_components_kept': 0,
            'component_sizes': [], 'fg_voxels': 0, 'total_voxels': total_voxels,
        }

    volume_patches = []
    npc_centers = []
    component_sizes = []
    fg_voxels = 0
    ps_z, ps_y, ps_x = patch_size

    for prop in regionprops(labeled):
        component_sizes.append(int(prop.area))
        fg_voxels += int(prop.area)  # avoids a second full-volume bool scan later
        if prop.area < min_npc_size:
            continue
        zc, yc, xc = prop.centroid
        npc_centers.append((zc, yc, xc))

        # Clamp instead of skip: NPCs near the volume edge (polar-facing disks)
        # produce ring morphology in en-face view. Skipping them teaches the
        # model only side-on appearance and causes high FN rates at the top
        # and bottom of the stack.
        z0 = int(np.clip(int(zc) - ps_z // 2, 0, max(0, Z - ps_z)))
        y0 = int(np.clip(int(yc) - ps_y // 2, 0, max(0, H - ps_y)))
        x0 = int(np.clip(int(xc) - ps_x // 2, 0, max(0, W - ps_x)))
        z1 = z0 + ps_z; y1 = y0 + ps_y; x1 = x0 + ps_x
        volume_patches.append((str(img_p), str(mask_p), z0, z1, y0, y1, x0, x1))

        if augment and jitter_range > 0:
            jz = rng.randint(-jitter_range, jitter_range + 1)
            jy = rng.randint(-jitter_range, jitter_range + 1)
            jx = rng.randint(-jitter_range, jitter_range + 1)
            jz0 = int(np.clip(z0 + jz, 0, max(0, Z - ps_z)))
            jy0 = int(np.clip(y0 + jy, 0, max(0, H - ps_y)))
            jx0 = int(np.clip(x0 + jx, 0, max(0, W - ps_x)))
            jz1 = jz0 + ps_z; jy1 = jy0 + ps_y; jx1 = jx0 + ps_x
            if jz0 <= zc <= jz1 and jy0 <= yc <= jy1 and jx0 <= xc <= jx1:
                volume_patches.append(
                    (str(img_p), str(mask_p), jz0, jz1, jy0, jy1, jx0, jx1))

    del labeled  # free the 4-byte/voxel int32 array before negative sampling

    # Record the fg boundary before appending negatives.
    n_fg_patches = len(volume_patches)

    # Negative samples
    if add_negative_samples and volume_patches:
        n_neg = int(n_fg_patches * negative_ratio)
        n_hard = int(n_neg * 0.6)
        n_rand = n_neg - n_hard

        hard_count, attempts = 0, 0
        while hard_count < n_hard and attempts < n_hard * 10 and npc_centers:
            attempts += 1
            zc, yc, xc = npc_centers[rng.randint(len(npc_centers))]
            off = rng.randint(60, 120, 3) * rng.choice([-1, 1], 3)
            nz0 = int(np.clip(int(zc+off[0])-ps_z//2, 0, max(0, Z-ps_z)))
            ny0 = int(np.clip(int(yc+off[1])-ps_y//2, 0, max(0, H-ps_y)))
            nx0 = int(np.clip(int(xc+off[2])-ps_x//2, 0, max(0, W-ps_x)))
            nz1 = nz0+ps_z; ny1 = ny0+ps_y; nx1 = nx0+ps_x
            if mask_vol[nz0:nz1, ny0:ny1, nx0:nx1].sum() < min_npc_size * 0.1:
                volume_patches.append(
                    (str(img_p), str(mask_p), nz0, nz1, ny0, ny1, nx0, nx1))
                hard_count += 1

        rand_count, attempts = 0, 0
        while rand_count < n_rand and attempts < n_rand * 10:
            attempts += 1
            rz0 = rng.randint(0, max(1, Z-ps_z))
            ry0 = rng.randint(0, max(1, H-ps_y))
            rx0 = rng.randint(0, max(1, W-ps_x))
            rz1 = rz0+ps_z; ry1 = ry0+ps_y; rx1 = rx0+ps_x
            if mask_vol[rz0:rz1, ry0:ry1, rx0:rx1].sum() < min_npc_size * 0.1:
                volume_patches.append(
                    (str(img_p), str(mask_p), rz0, rz1, ry0, ry1, rx0, rx1))
                rand_count += 1

        del mask_vol

    fg_flags = ([True] * n_fg_patches
                + [False] * (len(volume_patches) - n_fg_patches))

    if max_patches_per_volume and len(volume_patches) > max_patches_per_volume:
        sub_idx = rng.choice(len(volume_patches), max_patches_per_volume,
                             replace=False)
        volume_patches = [volume_patches[i] for i in sub_idx]
        fg_flags = [fg_flags[i] for i in sub_idx]

    return {
        'img_p': str(img_p), 'mask_p': str(mask_p),
        'stats': stats, 'patches': volume_patches, 'fg_flags': fg_flags,
        'n_components_total': n_npc,
        'n_components_kept': len(npc_centers),
        'component_sizes': component_sizes,
        'fg_voxels': fg_voxels,
        'total_voxels': total_voxels,
    }


# Conditioning input ~= noise_metric * NOISE_COND_SCALE. The metric is a
# brightness-invariant inverse-SNR (~0.1 clean to ~2 noisy), already in a
# usable range, so the scale is 1.0; explicit --noise-levels overrides
# should be picked on a comparable scale (e.g. 0 clean, ~1 noisy).
NOISE_COND_SCALE = 1.0
_IMMERKAER_L = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
                        dtype=np.float32)


def _immerkaer_sigma(slices_raw):
    """Immerkaer (1996) Laplacian high-frequency noise sigma in RAW units,
    median over a few slices. Robust to smooth structure (the Laplacian
    kills low frequencies)."""
    c = np.sqrt(np.pi / 2) / 6.0
    vals = [c * float(np.mean(np.abs(
        convolve(s.astype(np.float32), _IMMERKAER_L, mode='reflect'))))
        for s in slices_raw]
    return float(np.median(vals)) if vals else 0.0


def compute_volume_stats_streaming(path, sample_slices=40):
    """
    Compute normalisation stats by sampling slices from a large volume
    instead of loading the whole thing (~400 x 2720 x 2720 = ~6 GB float32).
    Returns dict with lo, hi, mean, std, noise_sigma.
    """
    with tifffile.TiffFile(str(path)) as tf:
        n_pages = len(tf.pages)
        step = max(1, n_pages // sample_slices)
        indices = list(range(0, n_pages, step))[:sample_slices]
        samples = []
        for i in indices:
            s = tf.pages[i].asarray().astype(np.float32)
            samples.append(s)
    stack = np.stack(samples)
    lo, hi = np.percentile(stack, (0.5, 99.5))
    clipped = np.clip(stack, lo, hi)
    # Relative noise = raw high-freq noise sigma / a noise-insensitive
    # brightness scale (median of the clipped image). The median is robust
    # to zero-mean noise, so noisier acquisitions score higher even when
    # noise dominates the dynamic range.
    signal = max(abs(float(np.median(clipped))), 1e-6)
    return {
        'lo': float(lo), 'hi': float(hi),
        'mean': float(clipped.mean()),
        'std': float(clipped.std() + 1e-6),
        'noise_sigma': _immerkaer_sigma(stack) / signal,
    }


def clahe_weak_signal_filter(img, tophat_size=17, clip_limit=0.01,
                             gate_window=5, gate_pct=20.0):
    """
    CLAHE that does not amplify flat background into spurious structure.
      1. white top-hat (~2x NPC radius) strips slow pan-stain background;
      2. per-slice CLAHE on the band-passed, [0,1]-scaled image;
      3. local-variance soft gate damps flat-but-noisy regions toward 0;
      4. median/MAD normalization (robust to CLAHE's widened background).
    """
    img = white_tophat(img, size=tophat_size).astype(np.float32)
    lo, hi = np.percentile(img, (0.5, 99.5))
    span = hi - lo
    img = (np.clip(img, lo, hi) - lo) / span if span > 0 else np.zeros_like(img)

    out = np.empty_like(img, dtype=np.float32)
    for z in range(img.shape[0]):
        out[z] = equalize_adapthist(img[z].astype(np.float64),
                                    clip_limit=clip_limit).astype(np.float32)

    mean = uniform_filter(out, gate_window)
    mean2 = uniform_filter(out * out, gate_window)
    local_std = np.sqrt(np.clip(mean2 - mean * mean, 0, None))
    tau = np.percentile(local_std, gate_pct)
    scale = local_std.std() * 0.5 + 1e-6
    gate = 1.0 / (1.0 + np.exp(-(local_std - tau) / scale))

    med = np.median(out)
    mad = np.median(np.abs(out - med)) * 1.4826 + 1e-6
    out = (out - med) / mad
    return (out * gate).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NPCDataset3D(Dataset):
    def __init__(self, root, patch_size=96, augment=False,
                 max_patches_per_volume=None, min_npc_size=1000,
                 add_negative_samples=True, negative_ratio=0.3,
                 jitter_range=24, use_elastic=False,
                 so3_rotation_prob=0.75,
                 rescale_range=(0.85, 1.18),
                 rescale_prob=0.5,
                 z_scale_range=(0.5, 2.0),
                 elastic_spacing=32, elastic_sigma=4.0,
                 elastic_mode='trilinear',
                 use_clahe=False, clahe_tophat_size=17,
                 clahe_clip_limit=0.01,
                 noise_aug=False, noise_aug_prob=0.5,
                 noise_aug_gain=0.2, use_bias_field=False,
                 bias_field_prob=0.5, bias_field_max=0.3,
                 extraction_workers=1,
                 preload_volumes=False,
                 dist_sigma=4.0):
        self.root = Path(root)
        self.min_npc_size = min_npc_size
        self.dist_sigma = float(dist_sigma)
        # Accept .tif and .tiff; pair by sorted order within each subdir.
        self.img_paths = sorted(list((self.root / "images").glob("*.tif"))
                                + list((self.root / "images").glob("*.tiff")))
        self.mask_paths = sorted(list((self.root / "masks").glob("*.tif"))
                                 + list((self.root / "masks").glob("*.tiff")))
        assert len(self.img_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.img_paths)} images vs {len(self.mask_paths)} masks"

        self.patch_size = (patch_size, patch_size, patch_size)
        self.augment = augment
        self.use_elastic = use_elastic and augment
        self.so3_rotation_prob = so3_rotation_prob if augment else 0.0
        self.rescale_range = rescale_range if augment else None
        self.rescale_prob = rescale_prob if augment else 0.0
        self.z_scale_range = z_scale_range if augment else None
        self.elastic_spacing = elastic_spacing
        self.elastic_sigma = elastic_sigma
        self.elastic_mode = elastic_mode
        self.use_clahe = use_clahe
        self.clahe_tophat_size = clahe_tophat_size
        self.clahe_clip_limit = clahe_clip_limit
        self.noise_aug = noise_aug and augment
        self.noise_aug_prob = noise_aug_prob
        self.noise_aug_gain = noise_aug_gain
        self.use_bias_field = use_bias_field and augment
        self.bias_field_prob = bias_field_prob
        self.bias_field_max = bias_field_max
        self.patch_list = []
        self._patch_fg_flags = []
        self._vol_stats = {}
        self._vol_shapes = {}
        self._vol_meta = []
        self._noise_max = 1.0

        # Cache key from every extraction parameter + file mtimes/sizes,
        # so a changed param or touched file forces a rebuild.
        img_mtimes = sorted((str(p), p.stat().st_mtime, p.stat().st_size)
                            for p in self.img_paths)
        mask_mtimes = sorted((str(p), p.stat().st_mtime, p.stat().st_size)
                             for p in self.mask_paths)
        cache_key_data = {
            'img_files': img_mtimes,
            'mask_files': mask_mtimes,
            'patch_size': self.patch_size,
            'min_npc_size': min_npc_size,
            'augment': bool(augment),
            'jitter_range': jitter_range,
            'add_negative_samples': bool(add_negative_samples),
            'negative_ratio': negative_ratio if add_negative_samples else 0,
            'max_patches_per_volume': max_patches_per_volume,
            'extractor_version': 8,
        }
        cache_key = hashlib.sha256(
            repr(sorted(cache_key_data.items())).encode()).hexdigest()[:16]
        cache_dir = self.root / "_patch_cache"
        cache_path = cache_dir / f"patches_{cache_key}.pkl"

        if cache_path.exists():
            print(f"[cache] Loading patch list from {cache_path}")
            t0 = time.time()
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            self.patch_list = cached['patches']
            self._patch_fg_flags = cached['patch_fg_flags']
            self._vol_stats = cached['vol_stats']
            self._vol_meta = cached['vol_meta']
            print(f"[cache] Loaded {len(self.patch_list)} patches in "
                  f"{time.time()-t0:.1f}s. To rebuild, delete {cache_path}.")
        else:
            print(f"Extracting patches (size={patch_size}, augment={augment}, "
                  f"min_npc={min_npc_size}, jitter={jitter_range})...")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._run_extraction(
                min_npc_size=min_npc_size,
                augment=augment, jitter_range=jitter_range,
                add_negative_samples=add_negative_samples,
                negative_ratio=negative_ratio,
                max_patches_per_volume=max_patches_per_volume,
                num_workers=extraction_workers)
            tmp_path = cache_path.with_suffix('.pkl.tmp')
            with open(tmp_path, 'wb') as f:
                pickle.dump({
                    'patches': self.patch_list,
                    'patch_fg_flags': self._patch_fg_flags,
                    'vol_stats': self._vol_stats,
                    'vol_meta': self._vol_meta,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.rename(cache_path)
            print(f"[cache] Wrote {cache_path}")

        # Cache (Z, H, W) per image so __getitem__ opens each file once.
        for p in self.img_paths:
            with tifffile.TiffFile(str(p)) as tf:
                self._vol_shapes[str(p)] = (len(tf.pages),
                                            *tf.pages[0].shape)

        vals = [self._noise_value(p) for p in self.img_paths]
        self._noise_max = max(vals) if vals else 1.0

        self._fg_indices = [i for i, fg in enumerate(self._patch_fg_flags) if fg]
        self._bg_indices = [i for i, fg in enumerate(self._patch_fg_flags) if not fg]

        self._shm_vols = {}
        self._mmap_vols = {}
        if preload_volumes:
            self._preload_volumes()

    def _preload_volumes(self, n_threads=16):
        """Maximally efficient volume loading strategy:
        1. Fill /dev/shm as far as it allows (partial, smallest volumes first).
        2. For the remainder, memmap the TIF files directly — uncompressed TIFs
           are a flat binary blob after the header, so memmap gives OS-page-cache
           sharing across all 22+ DataLoader workers with no extra disk copy.
        Workers that hit a shm entry do zero-copy tensor slicing; workers that
        hit a mmap entry do a numpy slice (kernel handles caching and sharing)."""
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        pairs = list(zip(self.img_paths, self.mask_paths))
        pair_bytes = [(p[0].stat().st_size + p[1].stat().st_size, p)
                      for p in pairs]
        pair_bytes.sort(key=lambda x: x[0])  # smallest first → more fit in shm

        shm_free = shutil.disk_usage('/dev/shm').free
        shm_budget = shm_free * 0.85  # leave 15% headroom
        shm_pairs, mmap_pairs = [], []
        used = 0
        for sz, pair in pair_bytes:
            if used + sz <= shm_budget:
                shm_pairs.append(pair)
                used += sz
            else:
                mmap_pairs.append(pair)

        total_bytes = sum(s for s, _ in pair_bytes)
        print(f"  [preload] {total_bytes/1e9:.1f} GB total, "
              f"{shm_free/1e9:.1f} GB /dev/shm free. "
              f"SHM: {len(shm_pairs)}/{len(pairs)} vols "
              f"({used/1e9:.1f} GB), "
              f"mmap: {len(mmap_pairs)} vols.")

        if shm_pairs:
            shm_paths = [p for pair in shm_pairs for p in pair]
            print(f"  Preloading {len(shm_pairs)} volumes ({used/1e9:.1f} GB) "
                  f"into shared memory...")

            def _load_shm(p):
                arr = tifffile.imread(str(p))
                t = torch.from_numpy(np.ascontiguousarray(arr))
                t.share_memory_()
                return str(p), t

            with ThreadPoolExecutor(max_workers=min(n_threads, len(shm_paths))) as ex:
                for path, tensor in tqdm(
                        ex.map(_load_shm, shm_paths), total=len(shm_paths),
                        desc="  Preloading SHM"):
                    self._shm_vols[path] = tensor
            print("  Shared memory preload complete.")

        if mmap_pairs:
            mmap_paths = [p for pair in mmap_pairs for p in pair]
            n_ok, n_fail = 0, 0
            for p in mmap_paths:
                try:
                    arr = tifffile.memmap(str(p), mode='r')
                    self._mmap_vols[str(p)] = arr
                    n_ok += 1
                except Exception:
                    n_fail += 1  # silently fall back to per-getitem TIF reads
            status = f"{n_ok} ok"
            if n_fail:
                status += f", {n_fail} failed (will read TIF per-item)"
            print(f"  mmap: {status}.")

    def _noise_value(self, img_p):
        """Per-volume conditioning scalar from Immerkaer noise sigma estimate.
        Higher = noisier acquisition."""
        stats = self._vol_stats.get(str(img_p))
        sig = stats.get('noise_sigma') if stats else None
        return float(sig * NOISE_COND_SCALE) if sig is not None else 0.0

    def _normalize_patch(self, img, img_p):
        """Percentile-clip + z-score, or the CLAHE weak-signal path."""
        if self.use_clahe:
            return clahe_weak_signal_filter(
                img, tophat_size=self.clahe_tophat_size,
                clip_limit=self.clahe_clip_limit)
        stats = self._vol_stats.get(str(img_p))
        if stats:
            img = np.clip(img, stats['lo'], stats['hi'])
            return ((img - stats['mean']) / stats['std']).astype(np.float32)
        lo, hi = np.percentile(img, (0.5, 99.5)).astype(np.float32)
        img = np.clip(img, lo, hi)
        return ((img - img.mean()) / (img.std() + 1e-6)).astype(np.float32)

    def _run_extraction(self, min_npc_size, augment, jitter_range,
                        add_negative_samples, negative_ratio,
                        max_patches_per_volume, num_workers):
        """Per-volume extraction, parallel via spawn Pool if num_workers>1.
        Each worker gets a distinct seed so jitter offsets differ."""
        worker_args = [
            (self.img_paths[i], self.mask_paths[i], self.patch_size,
             min_npc_size, augment, jitter_range,
             add_negative_samples, negative_ratio, max_patches_per_volume,
             42 + i)
            for i in range(len(self.img_paths))
        ]

        if num_workers and num_workers > 1:
            n_workers_actual = min(num_workers, len(worker_args))
            print(f"  Parallel extraction across {n_workers_actual} workers...")
            try:
                ctx = mp.get_context('spawn')
                with ctx.Pool(n_workers_actual) as pool:
                    results = list(tqdm(
                        pool.imap_unordered(_extract_volume_patches,
                                            worker_args),
                        total=len(worker_args), desc="Volumes"))
            except (PicklingError, AttributeError, RuntimeError) as e:
                print(f"  Parallel extraction failed "
                      f"({type(e).__name__}); falling back to serial.")
                results = [_extract_volume_patches(a)
                           for a in tqdm(worker_args, desc="Volumes")]
        else:
            results = [_extract_volume_patches(a)
                       for a in tqdm(worker_args, desc="Volumes")]

        results.sort(key=lambda r: r['img_p'])  # imap_unordered scrambles
        for r in results:
            self._vol_stats[r['img_p']] = r['stats']
            self.patch_list.extend(r['patches'])
            self._patch_fg_flags.extend(r['fg_flags'])
            self._vol_meta.append({
                'img_p': r['img_p'],
                'n_components_total': r['n_components_total'],
                'n_components_kept': r['n_components_kept'],
                'component_sizes': r['component_sizes'],
                'fg_voxels': r['fg_voxels'],
                'total_voxels': r['total_voxels'],
            })
        print(f"  Total patches: {len(self.patch_list)}")

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        img_p, mask_p, z0, z1, y0, y1, x0, x1 = self.patch_list[idx]

        # Size the source-read margin to the chosen augmentation: cube-only
        # needs none; SO(3) needs the sqrt(3) diagonal; zoom-out scales it up.
        ps = self.patch_size[0]
        plan = self._sample_augmentation_plan() if self.augment else None
        if plan is None or plan['mode'] in ('none', 'cube'):
            margin_each = 0
        else:
            effective_scale = min(plan['scale'], plan.get('scale_z', plan['scale']), 1.0)
            margin_each = int(np.ceil(
                (ps / 2) * (np.sqrt(3) / effective_scale - 1.0)))
            margin_each = max(margin_each, 0)

        Z_full, H_full, W_full = self._vol_shapes[str(img_p)]

        ez0 = max(0, z0 - margin_each); ez1 = min(Z_full, z1 + margin_each)
        ey0 = max(0, y0 - margin_each); ey1 = min(H_full, y1 + margin_each)
        ex0 = max(0, x0 - margin_each); ex1 = min(W_full, x1 + margin_each)

        img_key, mask_key = str(img_p), str(mask_p)
        if img_key in self._shm_vols:
            img = self._shm_vols[img_key][ez0:ez1, ey0:ey1, ex0:ex1].numpy().astype(np.float32)
            mask = self._shm_vols[mask_key][ez0:ez1, ey0:ey1, ex0:ex1].numpy().astype(np.float32)
        elif img_key in self._mmap_vols:
            img = np.array(self._mmap_vols[img_key][ez0:ez1, ey0:ey1, ex0:ex1], dtype=np.float32)
            mask = np.array(self._mmap_vols[mask_key][ez0:ez1, ey0:ey1, ex0:ex1], dtype=np.float32)
        else:
            with tifffile.TiffFile(str(img_p)) as tf:
                img = np.stack([tf.pages[z].asarray()[ey0:ey1, ex0:ex1]
                                for z in range(ez0, ez1)])
            with tifffile.TiffFile(str(mask_p)) as tf:
                mask = np.stack([tf.pages[z].asarray()[ey0:ey1, ex0:ex1]
                                 for z in range(ez0, ez1)])

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        # Drop components < min_npc_size so small specks never supervise.
        if self.min_npc_size and self.min_npc_size > 0:
            mbin = mask > 0
            if mbin.any():
                lbl, n = label_3d(mbin)
                if n > 0:
                    sizes = np.bincount(lbl.ravel())
                    too_small = sizes < self.min_npc_size
                    too_small[0] = False
                    if too_small.any():
                        mask[too_small[lbl]] = 0.0

        img = self._normalize_patch(img, img_p)

        # Noise-matched augmentation: raise a cleaner patch's effective noise
        # toward the dataset's noisiest tier and set the conditioning scalar
        # to match, so the FiLM head learns a consistent clean<->noisy axis.
        noise_val = self._noise_value(img_p)
        if self.noise_aug and np.random.rand() < self.noise_aug_prob \
                and self._noise_max > noise_val:
            r_target = float(np.random.uniform(noise_val, self._noise_max))
            # add_std is in z-scored/MAD image units (~unit std); gain sets
            # how much noise one unit of conditioning delta corresponds to.
            add_std = self.noise_aug_gain * np.sqrt(
                max(0.0, r_target ** 2 - noise_val ** 2))
            if add_std > 0:
                img = img + np.random.normal(
                    0, add_std, img.shape).astype(np.float32)
            noise_val = r_target

        if self.augment:
            img, mask = self._augment(img, mask, plan)

        # Center-crop back to patch_size, padding if the read was edge-clipped.
        if img.shape != (ps, ps, ps):
            cz = (img.shape[0] - ps) // 2
            cy = (img.shape[1] - ps) // 2
            cx = (img.shape[2] - ps) // 2
            def _crop_or_pad(a, target):
                D, H, W = a.shape
                pads = []
                slices = []
                for dim, t in zip([D, H, W], target):
                    if dim >= t:
                        s = (dim - t) // 2
                        slices.append(slice(s, s + t)); pads.append((0, 0))
                    else:
                        slices.append(slice(0, dim))
                        deficit = t - dim
                        pads.append((deficit // 2, deficit - deficit // 2))
                a = a[slices[0], slices[1], slices[2]]
                if any(p != (0, 0) for p in pads):
                    a = np.pad(a, pads, mode='reflect')
                return a
            img = _crop_or_pad(img, (ps, ps, ps))
            mask = _crop_or_pad(mask, (ps, ps, ps))

        binary = (mask > 0).astype(np.float32)
        wmap = np.ones_like(binary, dtype=np.float32)
        # Gaussian-blurred binary as topology-agnostic distance heatmap target.
        # A face-on disk, 2-ring, and 3-ring NPC all produce a similar centroid-
        # centred response, so the dist head gets a consistent shape prior
        # regardless of which view the patch samples.
        dist_arr = gaussian_filter(binary, sigma=self.dist_sigma)
        if dist_arr.max() > 0:
            dist_arr = (dist_arr / dist_arr.max()).astype(np.float32)
        return (torch.from_numpy(img.astype(np.float32))[None],
                torch.from_numpy(binary)[None],
                torch.from_numpy(wmap)[None],
                torch.tensor(noise_val, dtype=torch.float32),
                torch.from_numpy(dist_arr)[None])

    def _sample_augmentation_plan(self):
        """
        Make the per-patch augmentation choices once, up front. Returns a
        plan dict that __getitem__ uses to size the source-read margin
        and that _augment later consumes to apply the decided transforms
        without re-randomizing.

        Sampling the plan first lets us avoid reading more source than
        the chosen augmentations actually need. Cube rotations need no
        margin at all; SO(3) rotations need the diagonal margin; rescale
        with scale<1 (zoom-out) adds proportional margin.
        """
        if not self.augment:
            return {'mode': 'none', 'R': None, 'scale': 1.0, 'scale_z': 1.0,
                    'do_elastic': False}

        # Decide whether to apply rescale (only meaningful if a rescale
        # range is configured)
        do_rescale = (self.rescale_range is not None
                      and np.random.rand() < self.rescale_prob)
        # Sampling matches the original _augment logic exactly so behavior
        # is unchanged -- we just compute the plan first instead of inline.
        if do_rescale or np.random.rand() < self.so3_rotation_prob:
            if np.random.rand() < self.so3_rotation_prob:
                R = _random_rotation_matrix_so3()
                mode = 'so3'
            else:
                R = np.array(CUBE_ROTATIONS[np.random.randint(24)],
                             dtype=np.float32)
                mode = 'cube_via_affine'
            if do_rescale:
                lo, hi = self.rescale_range
                scale = float(np.random.uniform(lo, hi))
            else:
                scale = 1.0
            # Independent Z scale for anisotropic disk-orientation augmentation.
            # scale_z > scale compresses Z (face-on ring); scale_z < scale
            # stretches Z (side-on 3-dot morphology). Applied every time we
            # are already in affine mode, so the model sees the full aspect-ratio
            # space even when no XY rescale was sampled.
            if self.z_scale_range is not None:
                zlo, zhi = self.z_scale_range
                scale_z = float(np.random.uniform(zlo, zhi))
            else:
                scale_z = scale
        else:
            R = CUBE_ROTATIONS[np.random.randint(24)]
            mode = 'cube'
            scale = 1.0
            scale_z = 1.0

        do_elastic = self.use_elastic and (np.random.rand() > 0.5)
        return {'mode': mode, 'R': R, 'scale': scale, 'scale_z': scale_z,
                'do_elastic': do_elastic}

    def _augment(self, img, mask, plan):
        """Apply augmentation according to the pre-sampled plan (the plan
        was used to size the source-read margin, so it must not change)."""
        # --- Orientation augmentation ---------------------------------------
        # Prefer arbitrary SO(3) rotations over the 24 cube rotations because
        # cube rotations only permute axes — they cannot turn a side-on NPC
        # into a top-down (en face) one.
        mode = plan['mode']
        if mode == 'so3' or mode == 'cube_via_affine':
            img, mask = apply_affine_3d(img, mask, R=plan['R'],
                                        scale=plan['scale'],
                                        scale_z=plan.get('scale_z'))
        elif mode == 'cube':
            img = apply_cube_rotation(img, plan['R'])
            mask = apply_cube_rotation(mask, plan['R'])
        # else mode == 'none', do nothing

        # --- Elastic, intensity --------------------------------------------
        if plan['do_elastic']:
            img, mask = elastic_deform_3d(
                img, mask,
                control_spacing=self.elastic_spacing,
                sigma=self.elastic_sigma,
                mode=self.elastic_mode)

        # Bias-field (illumination/staining inhomogeneity) before the
        # point intensity augs.
        if self.use_bias_field and np.random.rand() < self.bias_field_prob:
            img = random_bias_field_3d(img, max_coeff=self.bias_field_max)

        # Broader intensity / gamma augmentation: dim-NPC robustness needs
        # the model to see NPCs across a real range of relative brightness,
        # not just +/- 15%. Multiplicative scaling and gamma jitter applied
        # in the standardized-intensity space (img is roughly zero-mean unit-
        # std at this point), so we work on the signed value.
        if np.random.rand() > 0.5:
            img = img * np.random.uniform(0.6, 1.4) + np.random.uniform(-0.15, 0.15)
        if np.random.rand() > 0.5:
            # Signed gamma: preserves sign, applies gamma to magnitude.
            gamma = np.random.uniform(0.7, 1.4)
            img = np.sign(img) * (np.abs(img) ** gamma)
        # Additive noise is handled by the noise-matched augmentation in
        # __getitem__ (which also updates the conditioning scalar), so it is
        # intentionally not duplicated here. Per-patch Gaussian smoothing was
        # removed — it blurred the high-frequency contrast that distinguishes
        # dim NPCs from background texture.
        return img.astype(np.float32), mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred_mask, target_mask, threshold=0.5):
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    target_binary = (target_mask > 0.5).float()
    intersection = (pred_binary * target_binary).sum()
    dice = (2 * intersection + 1) / (pred_binary.sum() + target_binary.sum() + 1)
    tp = intersection
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {
        'dice': dice.item(), 'precision': precision.item(),
        'recall': recall.item(),
        'fg_pred': pred_binary.mean().item(),
        'fg_target': target_binary.mean().item(),
    }


def _gaussian_window(ps, device):
    c = (ps - 1) / 2.0
    coords = torch.arange(ps, device=device, dtype=torch.float32) - c
    g = torch.exp(-(coords ** 2) / (2 * (ps * 0.25) ** 2))
    return g[:, None, None] * g[None, :, None] * g[None, None, :]


@torch.inference_mode()
def sliding_window_probs(model, img, patch_size, stride, device, amp,
                         noise=None):
    """Gaussian-blended sliding-window sigmoid probs over a normalized
    (Z,H,W) numpy volume. Pads up to patch_size per axis if needed. If a
    noise scalar is given it is passed to the model's FiLM conditioning."""
    ps = patch_size
    Z, H, W = img.shape
    pad = ((0, max(0, ps - Z)), (0, max(0, ps - H)), (0, max(0, ps - W)))
    if any(b for _, b in pad):
        img = np.pad(img, pad, mode='reflect')
    Zp, Hp, Wp = img.shape
    use_amp = amp and device.type == 'cuda'
    acc = torch.zeros((Zp, Hp, Wp), device=device)
    wsum = torch.zeros((Zp, Hp, Wp), device=device)
    gw = _gaussian_window(ps, device)
    img_t = torch.from_numpy(img).to(device)
    cond = (torch.tensor([noise], device=device, dtype=torch.float32)
            if noise is not None else None)

    def starts(n):
        if n <= ps:
            return [0]
        s = list(range(0, n - ps + 1, stride))
        if s[-1] != n - ps:
            s.append(n - ps)
        return s

    for z in starts(Zp):
        for y in starts(Hp):
            for x in starts(Wp):
                patch = img_t[z:z+ps, y:y+ps, x:x+ps][None, None]
                with autocast(device_type='cuda', enabled=use_amp):
                    logit, _, _ = model(patch, cond)
                prob = torch.sigmoid(logit.float())[0, 0]
                acc[z:z+ps, y:y+ps, x:x+ps] += prob * gw
                wsum[z:z+ps, y:y+ps, x:x+ps] += gw
    prob = (acc / wsum.clamp_min(1e-6)).cpu().numpy()
    return prob[:Z, :H, :W]


def _tta_flip_probs(model, img, patch_size, stride, device, amp, noise):
    """Average sliding-window probs over the 8 axis-flip orientations
    (exact, interpolation-free TTA) to exploit NPC flip symmetry."""
    acc = None
    for fz in (False, True):
        for fy in (False, True):
            for fx in (False, True):
                axes = tuple(a for a, f in
                             zip((0, 1, 2), (fz, fy, fx)) if f)
                v = np.flip(img, axes).copy() if axes else img
                p = sliding_window_probs(model, v, patch_size, stride,
                                         device, amp, noise)
                if axes:
                    p = np.flip(p, axes).copy()
                acc = p if acc is None else acc + p
    return acc / 8.0


def _group_to_npc_level(binary_mask, radius):
    """Dilate binary mask and re-label to merge per-NPC ring components.

    NPCs appear as 1-3 separate connected rings depending on the viewing
    angle. Dilating by `radius` voxels bridges the gap between co-NPC rings
    so they merge into one label under label_3d. The labels are then
    restricted back to the original (undilated) voxels so no new voxels are
    added. This gives NPC-level instance labels suitable for IoU matching.
    """
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=np.int32), 0
    dilated = binary_dilation(binary_mask, iterations=radius)
    grouped, n = label_3d(dilated)
    npc_lbl = (grouped * binary_mask).astype(np.int32)
    # n_groups may over-count if dilation split a group at the boundary;
    # recount unique non-zero labels in the restricted mask.
    n_actual = int(npc_lbl.max())
    return npc_lbl, n_actual


def _detection_counts(pred_bin, gt_bin, min_size, iou_thresh,
                      npc_group_radius=0):
    """Per-NPC detection TP/FP/FN by connected-component IoU matching.

    When npc_group_radius > 0, both pred and GT binary masks are first
    dilated to merge rings of the same NPC into one instance before IoU
    matching. This fixes the v8 behaviour where a 3-ring NPC counted as 3
    separate GT objects and any topology mismatch caused spurious FP/FN.
    GT components below min_size are ignored; predicted components are all
    counted.
    """
    if npc_group_radius > 0:
        gt_lbl, n_gt = _group_to_npc_level(gt_bin, npc_group_radius)
        pred_lbl, n_pred = _group_to_npc_level(pred_bin, npc_group_radius)
    else:
        gt_lbl, n_gt = label_3d(gt_bin)
        pred_lbl, n_pred = label_3d(pred_bin)
    gt_sizes = np.bincount(gt_lbl.ravel(), minlength=n_gt + 1)
    pred_sizes = np.bincount(pred_lbl.ravel(), minlength=n_pred + 1)
    both = (gt_lbl > 0) & (pred_lbl > 0)
    matched_gt, matched_pred = set(), set()
    if both.any():
        key = pred_lbl[both].astype(np.int64) * (n_gt + 1) + gt_lbl[both]
        kc = np.bincount(key)
        for k in np.nonzero(kc)[0]:
            pid, gid = int(k // (n_gt + 1)), int(k % (n_gt + 1))
            if pid == 0 or gid == 0:
                continue
            inter = kc[k]
            union = pred_sizes[pid] + gt_sizes[gid] - inter
            if union > 0 and inter / union >= iou_thresh:
                matched_gt.add(gid)
                matched_pred.add(pid)
    valid_gt = [g for g in range(1, n_gt + 1) if gt_sizes[g] >= min_size]
    tp = sum(1 for g in valid_gt if g in matched_gt)
    fn = sum(1 for g in valid_gt if g not in matched_gt)
    fp = sum(1 for p in range(1, n_pred + 1) if p not in matched_pred)
    return tp, fp, fn


def _filter_small_components(mask_bin, min_size):
    if not min_size or min_size <= 0 or not mask_bin.any():
        return mask_bin
    lbl, n = label_3d(mask_bin)
    if n == 0:
        return mask_bin
    sizes = np.bincount(lbl.ravel())
    too_small = sizes < min_size
    too_small[0] = False
    if too_small.any():
        mask_bin = mask_bin & ~too_small[lbl]
    return mask_bin


@torch.inference_mode()
def evaluate_full_volume(model, val_ds, device, amp, patch_size, stride,
                         roi_size, max_volumes, threshold=0.5,
                         iou_thresh=0.5, tta=False, npc_group_radius=0):
    """Sliding-window inference over an ROI per validation volume, then
    stitched volume Dice/IoU and per-NPC detection F1. The ROI is centered
    on a foreground patch center so the metric reflects realistic mixed
    foreground/background tiling rather than NPC-centered patches."""
    by_vol = {}
    for rec in val_ds.patch_list:
        img_p, mask_p, z0, z1, y0, y1, x0, x1 = rec
        by_vol.setdefault((img_p, mask_p), []).append(
            ((z0 + z1) // 2, (y0 + y1) // 2, (x0 + x1) // 2))
    vols = list(by_vol.items())[:max_volumes]

    dices, ious = [], []
    tps = fps = fns = 0
    for (img_p, mask_p), centers in vols:
        Z, H, W = val_ds._vol_shapes[str(img_p)]
        cz, cy, cx = centers[len(centers) // 2]
        half = roi_size // 2
        rz0 = int(np.clip(cz - half, 0, max(0, Z - roi_size)))
        ry0 = int(np.clip(cy - half, 0, max(0, H - roi_size)))
        rx0 = int(np.clip(cx - half, 0, max(0, W - roi_size)))
        rz1, ry1, rx1 = min(Z, rz0 + roi_size), min(H, ry0 + roi_size), min(W, rx0 + roi_size)

        with tifffile.TiffFile(str(img_p)) as tf:
            img = np.stack([tf.pages[z].asarray()[ry0:ry1, rx0:rx1]
                            for z in range(rz0, rz1)]).astype(np.float32)
        with tifffile.TiffFile(str(mask_p)) as tf:
            mask = np.stack([tf.pages[z].asarray()[ry0:ry1, rx0:rx1]
                             for z in range(rz0, rz1)]).astype(np.float32)

        img = val_ds._normalize_patch(img, img_p)
        noise = val_ds._noise_value(img_p)
        if tta:
            prob = _tta_flip_probs(model, img, patch_size, stride,
                                   device, amp, noise)
        else:
            prob = sliding_window_probs(model, img, patch_size, stride,
                                        device, amp, noise)
        pred = prob > threshold
        gt = _filter_small_components(mask > 0.5, val_ds.min_npc_size)

        inter = float((pred & gt).sum())
        dices.append((2 * inter + 1) / (pred.sum() + gt.sum() + 1))
        ious.append((inter + 1) / (float((pred | gt).sum()) + 1))
        t, f, n = _detection_counts(pred, gt, val_ds.min_npc_size, iou_thresh,
                                    npc_group_radius=npc_group_radius)
        tps += t; fps += f; fns += n

    det_prec = tps / (tps + fps + 1e-6)
    det_rec = tps / (tps + fns + 1e-6)
    det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec + 1e-6)
    return {
        'vol_dice': float(np.mean(dices)) if dices else 0.0,
        'vol_iou': float(np.mean(ious)) if ious else 0.0,
        'det_precision': det_prec, 'det_recall': det_rec, 'det_f1': det_f1,
    }


@torch.inference_mode()
def _bytes_human(n):
    """Convert byte count to human-readable string."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def stage_to_local_disk(train_dir, val_dir, local_root,
                        headroom_fraction=0.15):
    """Copy train/ + val/ to local disk. Pre-flight free-space check;
    temp-then-rename writes (crash-safe); skips size-matched existing files.
    Returns updated (train_dir, val_dir)."""
    import shutil as _sh
    local_root = Path(local_root)
    local_train = local_root / "train"
    local_val = local_root / "val"

    # ---- Pre-flight: how much data do we need to copy, how much free? ----
    def _collect_sources(src, dst):
        """List (real_src, target) pairs and total bytes needing copy."""
        src, dst = Path(src), Path(dst)
        pairs = []
        bytes_to_copy = 0
        for sub in ('images', 'masks'):
            src_sub = src / sub
            if not src_sub.exists():
                continue
            for f in sorted(list(src_sub.glob("*.tif"))
                            + list(src_sub.glob("*.tiff"))):
                real_src = f.resolve()
                target = dst / sub / f.name
                src_size = real_src.stat().st_size
                # Skip if target exists with matching size (already done)
                if target.exists():
                    try:
                        if target.stat().st_size == src_size:
                            continue
                    except OSError:
                        pass
                pairs.append((real_src, target, src_size))
                bytes_to_copy += src_size
        return pairs, bytes_to_copy

    # Make destination subtrees so we can stat them
    for split_dst in (local_train, local_val):
        for sub in ('images', 'masks'):
            (split_dst / sub).mkdir(parents=True, exist_ok=True)

    train_pairs, train_bytes = _collect_sources(train_dir, local_train)
    val_pairs, val_bytes = _collect_sources(val_dir, local_val)
    total_bytes = train_bytes + val_bytes
    total_files = len(train_pairs) + len(val_pairs)

    # Disk free space at destination
    fs_stats = os.statvfs(str(local_root))
    free_bytes = fs_stats.f_bavail * fs_stats.f_frsize
    required_bytes = int(total_bytes * (1 + headroom_fraction))

    print(f"\n[stage] Copying data from NFS to {local_root}")
    print(f"[stage]   train: {train_dir} -> {local_train}")
    print(f"[stage]   val:   {val_dir}   -> {local_val}")
    print(f"[stage] To copy: {total_files} files, "
          f"{_bytes_human(total_bytes)}")
    print(f"[stage] Free space at destination: {_bytes_human(free_bytes)}")
    print(f"[stage] Required (with {int(headroom_fraction*100)}% headroom): "
          f"{_bytes_human(required_bytes)}")

    if total_files == 0:
        print("[stage] Nothing to copy (all files already staged).")
        return str(local_train), str(local_val)

    if free_bytes < required_bytes:
        raise RuntimeError(
            f"\n[stage] Not enough free space at {local_root}.\n"
            f"  Need:  {_bytes_human(required_bytes)} "
            f"({_bytes_human(total_bytes)} of data + "
            f"{int(headroom_fraction*100)}% headroom)\n"
            f"  Free:  {_bytes_human(free_bytes)}\n"
            f"  Short: {_bytes_human(required_bytes - free_bytes)}\n\n"
            f"Common fixes:\n"
            f"  - In your sbatch script, request more local scratch:\n"
            f"      #SBATCH --tmp={int(required_bytes/1e9)+5}G\n"
            f"  - Set SLURM_TMPDIR to a larger filesystem.\n"
            f"  - Use --stage-to-local /path/to/larger/disk explicitly.\n"
            f"  - Drop --stage-to-local entirely and run from NFS "
            f"(slower but won't fail).")

    # ---- Copy with temp-then-rename to make partial writes safe ----
    t0 = time.time()
    pbar = tqdm(total=total_bytes, unit='B', unit_scale=True,
                desc="[stage] copying")
    try:
        for real_src, target, sz in train_pairs + val_pairs:
            tmp_target = target.with_suffix(target.suffix + '.partial')
            # Clean up any stale .partial from a prior crashed run
            if tmp_target.exists():
                tmp_target.unlink()
            try:
                _sh.copy2(str(real_src), str(tmp_target))
            except OSError as e:
                # Clean up before re-raising so the next run starts fresh
                if tmp_target.exists():
                    try:
                        tmp_target.unlink()
                    except OSError:
                        pass
                raise RuntimeError(
                    f"[stage] Copy failed: {real_src.name}: {e}") from e
            # Atomic move: only after the full file is on disk
            tmp_target.rename(target)
            pbar.update(sz)
    finally:
        pbar.close()

    dt = time.time() - t0
    print(f"[stage] Done: {total_files} files, "
          f"{_bytes_human(total_bytes)} in {dt:.1f}s "
          f"({_bytes_human(total_bytes/max(dt,0.001))}/s)")
    return str(local_train), str(local_val)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
class StratifiedPatchSampler(Sampler):
    """Sample n_total patches per epoch with a fixed fg/bg class ratio.
    Both pools are sampled with replacement so the ratio holds even when
    one pool is smaller than its quota (e.g. few bg patches in a small val
    split). A fresh RNG is drawn each __iter__ call for epoch-to-epoch
    variety."""
    def __init__(self, fg_indices, bg_indices, n_total, fg_ratio=0.7):
        self.fg = np.array(fg_indices, dtype=np.int64)
        self.bg = np.array(bg_indices, dtype=np.int64)
        self.n_total = n_total
        self.fg_ratio = fg_ratio

    def __iter__(self):
        rng = np.random.default_rng()
        n_fg = round(self.n_total * self.fg_ratio)
        n_bg = self.n_total - n_fg
        indices = np.concatenate([
            rng.choice(self.fg, size=n_fg, replace=True),
            rng.choice(self.bg, size=n_bg, replace=True),
        ])
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.n_total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start GPU keepalive immediately so the cluster doesn't cancel the job
    # for GPU idleness during the staging / patch-extraction phases.
    gpu_proc = None
    if torch.cuda.is_available():
        gpu_proc = start_gpu_keepalive()

    # Auto-partition from a paired flat folder (if --train_dir not given).
    if args.paired_dir:
        root = Path(args.partition_root or args.paired_dir)
        args.train_dir, args.val_dir = [
            str(d) for d in partition_train_val_paired(
                args.paired_dir, root, val_fraction=args.val_fraction,
                image_marker=args.paired_image_suffix,
                mask_marker=args.paired_mask_suffix)]

    # Optional: stage data to local disk for faster reads during training.
    # The single biggest win when source files live on NFS scratch: a
    # 124-volume dataset copies in ~1-2 minutes, then every patch read
    # comes from local disk. The patch cache key uses SOURCE file
    # mtimes/sizes, so the cache stays valid across staging.
    if args.stage_to_local:
        stage_dir = args.stage_to_local
        if stage_dir == "auto":
            if args.paired_dir:
                stage_dir = str(Path(args.paired_dir) / "_local_stage")
            else:
                raise RuntimeError(
                    "--stage-to-local=auto needs --paired-dir to derive a "
                    "default location. Pass an explicit path instead.")
        args.train_dir, args.val_dir = stage_to_local_disk(
            args.train_dir, args.val_dir, stage_dir)

    writer = SummaryWriter(args.logdir)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n" + "="*60 + "\nLOADING DATA\n" + "="*60)

    train_ds = NPCDataset3D(
        args.train_dir, patch_size=args.patch_size, augment=True,
        max_patches_per_volume=args.max_patches_per_volume,
        min_npc_size=args.min_npc_size,
        add_negative_samples=args.add_negative_samples,
        negative_ratio=args.negative_ratio,
        jitter_range=args.jitter_range,
        use_elastic=True,
        z_scale_range=tuple(args.z_scale_range),
        use_clahe=args.clahe, clahe_tophat_size=args.clahe_tophat_size,
        clahe_clip_limit=args.clahe_clip_limit,
        noise_aug=args.noise_aug,
        noise_aug_gain=args.noise_aug_gain, use_bias_field=args.bias_field,
        bias_field_max=args.bias_field_max,
        extraction_workers=args.extraction_workers,
        preload_volumes=args.preload_volumes,
        dist_sigma=args.dist_sigma)
    val_ds = NPCDataset3D(
        args.val_dir, patch_size=args.patch_size, augment=False,
        min_npc_size=args.min_npc_size, add_negative_samples=False,
        use_clahe=args.clahe, clahe_tophat_size=args.clahe_tophat_size,
        clahe_clip_limit=args.clahe_clip_limit,
        extraction_workers=args.extraction_workers,
        preload_volumes=args.preload_volumes,
        dist_sigma=args.dist_sigma)

    # Startup data audit: print per-split summary and sanity-check the
    # mask statistics. Useful when switching datasets, paired layouts,
    # or label-noise regimes -- a 5x change in foreground fraction
    # versus the previous run usually means something is wrong.
    # Uses _vol_meta cached during extraction so we don't repeat the
    # expensive connected-components pass.
    def _audit_split(name, ds):
        n_vols = len(ds.img_paths)
        n_patches = len(ds.patch_list)
        all_sizes = []
        fg_fracs = []
        for meta in ds._vol_meta:
            if meta['total_voxels'] > 0:
                fg_fracs.append(meta['fg_voxels'] / meta['total_voxels'])
            all_sizes.extend(meta['component_sizes'])
        n_components = len(all_sizes)
        if n_components > 0 and fg_fracs:
            sizes_arr = np.array(all_sizes)
            print(f"  [{name}] {n_vols} volumes, {n_patches} patches, "
                  f"{n_components} components total")
            print(f"  [{name}] foreground fraction per volume: "
                  f"min={min(fg_fracs):.4f}, "
                  f"median={float(np.median(fg_fracs)):.4f}, "
                  f"max={max(fg_fracs):.4f}")
            print(f"  [{name}] component size (voxels): "
                  f"min={int(sizes_arr.min())}, "
                  f"p25={int(np.percentile(sizes_arr, 25))}, "
                  f"median={int(np.median(sizes_arr))}, "
                  f"p75={int(np.percentile(sizes_arr, 75))}, "
                  f"max={int(sizes_arr.max())}")
        else:
            print(f"  [{name}] {n_vols} volumes, NO components found "
                  f"-- this is almost certainly a data problem")

    # Noise audit: group volumes by filename date prefix (YYYY-MM-DD) and
    # report the mean conditioning value, so the clean/noisy split can be
    # confirmed against the known acquisition dates.
    def _audit_noise(name, ds):
        groups = {}
        for p in ds.img_paths:
            key = Path(p).name[:10]
            groups.setdefault(key, []).append(ds._noise_value(p))
        if groups:
            print(f"  [{name}] noise by date prefix (conditioning value):")
            for key in sorted(groups):
                v = groups[key]
                print(f"    {key}: n={len(v)} mean={np.mean(v):.3f} "
                      f"min={min(v):.3f} max={max(v):.3f}")

    print("\nData audit:")
    _audit_split("train", train_ds)
    _audit_split("val", val_ds)
    _audit_noise("train", train_ds)
    _audit_noise("val", val_ds)

    nw = args.num_workers
    train_sampler = None
    train_shuffle = True
    if args.patches_per_epoch:
        train_shuffle = False
        if (args.fg_ratio > 0
                and train_ds._fg_indices and train_ds._bg_indices):
            train_sampler = StratifiedPatchSampler(
                train_ds._fg_indices, train_ds._bg_indices,
                n_total=args.patches_per_epoch,
                fg_ratio=args.fg_ratio)
            print(f"  Epoch subsampling: {args.patches_per_epoch} patches "
                  f"(stratified {args.fg_ratio:.0%} fg / "
                  f"{1-args.fg_ratio:.0%} bg; "
                  f"pools: {len(train_ds._fg_indices)} fg, "
                  f"{len(train_ds._bg_indices)} bg)")
        else:
            train_sampler = RandomSampler(
                train_ds, replacement=True,
                num_samples=args.patches_per_epoch)
            print(f"  Epoch subsampling: {args.patches_per_epoch} of "
                  f"{len(train_ds)} patches per epoch")
    def make_train_dl():
        return DataLoader(train_ds, batch_size=args.batch,
                          shuffle=train_shuffle, sampler=train_sampler,
                          num_workers=nw, pin_memory=True,
                          prefetch_factor=8 if nw > 0 else None,
                          persistent_workers=nw > 0, drop_last=True)
    train_dl = make_train_dl()
    val_sampler = None
    if args.val_patches and len(val_ds) > args.val_patches:
        val_sampler = RandomSampler(
            val_ds, replacement=False,
            num_samples=args.val_patches)
        print(f"  Val subsampling: {args.val_patches} of "
              f"{len(val_ds)} patches per validation")
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        sampler=val_sampler,
                        num_workers=nw, pin_memory=True,
                        prefetch_factor=8 if nw > 0 else None,
                        persistent_workers=nw > 0)

    stop_gpu_keepalive(gpu_proc)

    # Model
    model = UNet3D(in_c=1, base=args.base, depth=args.depth,
                   dropout=args.dropout,
                   deep_supervision=args.deep_supervision,
                   use_aaspp=args.aaspp,
                   noise_conditioning=args.noise_conditioning).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if hasattr(torch, 'compile') and args.compile:
        model = torch.compile(model, mode='reduce-overhead')

    ema = EMAModel(model, decay=args.ema_decay)

    # Smoke test
    if torch.cuda.is_available():
        ps = args.patch_size
        with autocast(device_type='cuda', enabled=args.amp):
            _out = model(torch.randn(1, 1, ps, ps, ps, device=device))
        del _out
        torch.cuda.empty_cache()
        print("GPU smoke-test passed")

    # Loss with scheduled hyperparameters. alpha/beta/w_bce/pos_weight
    # are each parsed from CLI as either a constant ("0.5") or a linear
    # ramp ("0.3:0.5:40"). The schedule is applied at the top of each
    # epoch via criterion.set_epoch(ep).
    alpha_sched = parse_schedule(args.tversky_alpha)
    beta_sched = parse_schedule(args.tversky_beta)
    w_bce_sched = parse_schedule(args.w_bce)
    pos_weight_sched = parse_schedule(args.pos_weight)
    w_cldice_sched = parse_schedule(args.w_cldice)

    loss_main = TverskyDiceLoss(
        w_bce_sched=w_bce_sched, w_tversky=args.w_dice,
        pos_weight_sched=pos_weight_sched,
        alpha_sched=alpha_sched, beta_sched=beta_sched,
        w_cldice_sched=w_cldice_sched, ft_gamma=args.ft_gamma,
        sce_alpha=args.sce_alpha, sce_beta=args.sce_beta,
        cldice_iters=args.cldice_iters,
        w_gce=args.w_gce, gce_q=args.gce_q).to(device)
    print(f"Loss: Focal-Tversky + SCE + GCE + clDice (scheduled)")
    print(f"  alpha:      {alpha_sched}")
    print(f"  beta:       {beta_sched}")
    print(f"  w_bce(SCE): {w_bce_sched}")
    print(f"  w_cldice:   {w_cldice_sched}")
    print(f"  w_gce:      {args.w_gce} (q={args.gce_q})")
    print(f"  pos_weight: {pos_weight_sched}")
    print(f"  w_tversky:  {args.w_dice}, ft_gamma: {args.ft_gamma}, "
          f"sce(a={args.sce_alpha},b={args.sce_beta})")

    loss_focal = FocalDiceLoss(
        w_focal=1.0, w_dice=args.w_dice).to(device)

    criterion = loss_main
    using_focal = False
    focal_recovery_done = False

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # LR warmup (LinearLR) then cosine annealing (SequentialLR). Warmup
    # stabilises early training with the 6-term compound loss. If
    # --lr-warmup-epochs 0, falls back to plain cosine annealing.
    wu = args.lr_warmup_epochs
    if wu > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=wu)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - wu), eta_min=args.min_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched], milestones=[wu])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler(enabled=args.amp)

    start_epoch, best_val_loss, best_val_dice, best_val_recall = 1, float('inf'), 0.0, 0.0
    global_step = 0

    if args.resume:
        from pathlib import Path as _P
        if not _P(args.resume).exists():
            print(f"\n--resume {args.resume} does not exist; starting fresh.")
        else:
            print(f"\nResuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            # Filter checkpoint tensors to those whose shape matches the
            # current model. This protects against architecture changes
            # like --base, --depth, --patch-size, --use-prosma-gates being
            # different from the run that produced the checkpoint.
            # strict=False alone does NOT handle shape mismatches; it only
            # tolerates missing/extra keys.
            model_sd = model.state_dict()
            ckpt_sd = ckpt.get('model', ckpt)  # raw state_dict also accepted
            compatible, skipped_shape, skipped_missing = {}, [], []
            for k, v in ckpt_sd.items():
                if k not in model_sd:
                    skipped_missing.append(k)
                elif tuple(v.shape) != tuple(model_sd[k].shape):
                    skipped_shape.append(
                        (k, tuple(v.shape), tuple(model_sd[k].shape)))
                else:
                    compatible[k] = v
            missing_in_ckpt = [k for k in model_sd if k not in ckpt_sd]
            model.load_state_dict(compatible, strict=False)
            print(f"  Loaded {len(compatible)}/{len(model_sd)} tensors.")
            if skipped_shape:
                print(f"  Skipped {len(skipped_shape)} tensors due to "
                      f"SHAPE MISMATCH (model architecture changed):")
                for k, s_ck, s_m in skipped_shape[:5]:
                    print(f"    {k}: ckpt {s_ck} vs model {s_m}")
                if len(skipped_shape) > 5:
                    print(f"    ... and {len(skipped_shape) - 5} more")
            if skipped_missing:
                print(f"  Skipped {len(skipped_missing)} tensors absent "
                      f"from current model.")
            if missing_in_ckpt:
                print(f"  Fresh init for {len(missing_in_ckpt)} tensors "
                      f"new to current model.")
            # Optimizer state only valid if the model is shape-compatible
            # (Adam's state tensors are sized to parameter tensors). If
            # any shapes changed, start the optimizer fresh.
            if 'opt' in ckpt and not skipped_shape:
                try:
                    opt.load_state_dict(ckpt['opt'])
                except (ValueError, KeyError):
                    print("  Optimizer state mismatch, starting fresh")
            elif skipped_shape:
                print("  Optimizer state discarded due to shape mismatches")
            if 'scheduler' in ckpt and not skipped_shape:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    print("  Scheduler state mismatch, starting fresh "
                          "(LR warmup may re-trigger)")
            elif not skipped_shape:
                # Old checkpoint with no scheduler state. If we're already
                # past the warmup phase, fast-forward the SequentialLR to
                # the correct cosine position so the LinearLR warmup does
                # not re-apply on top of already-trained weights.
                ckpt_ep = ckpt.get('epoch', 0)      # last completed epoch
                resume_ep = ckpt_ep + 1              # first epoch we'll train
                if (wu > 0 and resume_ep > wu
                        and isinstance(scheduler,
                                       torch.optim.lr_scheduler.SequentialLR)):
                    cosine_step = ckpt_ep - wu       # steps into cosine phase
                    scheduler.last_epoch = ckpt_ep
                    scheduler._schedulers[0].last_epoch = wu
                    scheduler._schedulers[1].last_epoch = cosine_step
                    print(f"  Fast-forwarded scheduler to ep {ckpt_ep} "
                          f"(cosine step {cosine_step}/"
                          f"{scheduler._schedulers[1].T_max}). "
                          f"LR warmup will not re-trigger.")
                else:
                    print("  Scheduler state not in checkpoint — warmup will "
                          "re-trigger if --lr-warmup-epochs > 0")
            if 'scaler' in ckpt and args.amp and not skipped_shape:
                scaler.load_state_dict(ckpt['scaler'])
            if 'ema' in ckpt and not skipped_shape:
                ema.shadow = ckpt['ema']
            elif 'ema' in ckpt:
                print("  EMA state discarded due to shape mismatches")
            # Resume training metadata only if the architecture matched
            # cleanly -- continuing the epoch counter on a different model
            # would be misleading.
            if not skipped_shape:
                start_epoch = ckpt.get('epoch', 0) + 1
                best_val_loss = ckpt.get('val_loss', float('inf'))
                best_val_dice = ckpt.get('val_dice', 0.0)
                best_val_recall = ckpt.get('val_recall', 0.0)
                global_step = ckpt.get('global_step', 0)
                focal_recovery_done = ckpt.get('focal_recovery_done', False)
            else:
                print("  Starting epoch=1, best metrics reset (architecture changed)")

    print("\n" + "="*60 + "\nTRAINING\n" + "="*60 + "\n")
    accumulation_steps = args.accumulation_steps

    for ep in range(start_epoch, args.epochs + 1):
        # Apply hyperparameter schedules for this (training) epoch.
        criterion.set_epoch(ep)
        if not using_focal:
            cv = criterion.current_values()
            print(f"  [schedule ep{ep}] alpha={cv['alpha']:.3f} "
                  f"beta={cv['beta']:.3f} w_bce={cv['w_bce']:.3f} "
                  f"w_cldice={cv['w_cldice']:.3f} "
                  f"pos_weight={cv['pos_weight']:.1f} "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

        model.train()
        train_loss = 0
        train_metrics = {'dice': 0, 'precision': 0, 'recall': 0}
        n_train_batches = 0   # batches that actually contributed to the mean
        nan_batches = 0       # batches skipped for non-finite loss
        t0 = time.time()
        pbar = tqdm(train_dl, desc=f"Ep {ep}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        accum_has_grad = False  # tracks whether backward() was called in current accumulation window

        # Diagnostic timing: separate "waiting for data" (dataloader-bound)
        # from "GPU compute" (model-bound). If data_wait_total >> gpu_total,
        # the dataloader is the bottleneck (raise --num-workers, simplify
        # augmentation, or stage data to local disk). If gpu_total dominates,
        # the model is the bottleneck (reduce --base or --patch-size, or
        # enable AMP). Printed at end of epoch.
        t_data_wait_total = 0.0
        t_gpu_total = 0.0
        t_last_batch_end = time.time()

        for batch_idx, (img, mask_tgt, wmap, noise, dist_tgt) in enumerate(pbar):
            # Time waiting for the dataloader to deliver this batch.
            # When workers > 0 and prefetch_factor > 1, this should be
            # near zero on a non-starved system.
            t_data_wait_total += time.time() - t_last_batch_end
            t_gpu_start = time.time()

            img = img.to(device, non_blocking=True)
            mask_tgt = mask_tgt.to(device, non_blocking=True)
            wmap = wmap.to(device, non_blocking=True)
            noise = noise.to(device, non_blocking=True)
            dist_tgt = dist_tgt.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=args.amp):
                mask_pred, dist_pred, ds_outputs = model(img, noise)

                seg_loss, _ = criterion(mask_pred, mask_tgt, weight_map=wmap)

                # Deep supervision loss — weighted sum of losses at each
                # decoder level, with exponentially decaying weights for
                # coarser levels (nnU-Net / Zeng et al. convention).
                # v9: pass pos_weight so coarse heads don't collapse to
                # all-background on this class-imbalanced target.
                ds_loss = torch.tensor(0.0, device=device)
                if ds_outputs:
                    ds_weight = 0.5
                    for ds_logits in ds_outputs[:-1]:  # skip finest (= main)
                        ds_bce = F.binary_cross_entropy_with_logits(
                            ds_logits, mask_tgt,
                            pos_weight=criterion.pos_weight,
                            reduction='mean')
                        ds_loss = ds_loss + ds_weight * ds_bce
                        ds_weight *= 0.5

                # Distance heatmap supervision: MSE between sigmoid(dist_pred)
                # and Gaussian-blurred binary target (topology-agnostic NPC
                # centroid heatmap). Trains the previously dead dist head.
                dist_loss = (F.mse_loss(torch.sigmoid(dist_pred.float()),
                                        dist_tgt.float())
                             if args.w_dist > 0
                             else torch.zeros((), device=device))

                loss = (seg_loss + ds_loss
                        + args.w_dist * dist_loss) / accumulation_steps

            # Guard against non-finite loss. A single NaN/Inf batch would
            # otherwise backprop NaN grads (GradScaler usually skips the step,
            # but clip_grad_norm_ on NaN grads and the polluted train_loss
            # readout are both undesirable). Skip this batch's backward so it
            # never touches grads; if it lands on an accumulation boundary,
            # still flush whatever good grads were accumulated this window.
            if not torch.isfinite(loss):
                nan_batches += 1
                if (batch_idx + 1) % accumulation_steps == 0:
                    if accum_has_grad:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scaler.step(opt)
                        scaler.update()
                        ema.update(model)
                    opt.zero_grad(set_to_none=True)
                    accum_has_grad = False
                t_gpu_total += time.time() - t_gpu_start
                t_last_batch_end = time.time()
                continue

            scaler.scale(loss).backward()
            accum_has_grad = True

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                accum_has_grad = False
                ema.update(model)

            train_loss += loss.item() * accumulation_steps
            n_train_batches += 1
            with torch.no_grad():
                metrics = compute_metrics(mask_pred, mask_tgt)
                for k in train_metrics:
                    train_metrics[k] += metrics[k]

            writer.add_scalar("train/loss", loss.item() * accumulation_steps, global_step)
            writer.add_scalar("train/dice", metrics['dice'], global_step)
            writer.add_scalar("train/recall", metrics['recall'], global_step)
            pbar.set_postfix(
                loss=f"{loss.item()*accumulation_steps:.4f}",
                dice=f"{metrics['dice']:.3f}",
                rec=f"{metrics['recall']:.3f}")
            global_step += 1

            # Close timing for this batch. CUDA synchronize ensures the
            # measured GPU time actually reflects compute, not enqueue.
            # We synchronize ONLY when computing the diagnostic, not every
            # iteration, to avoid hurting throughput in the typical case.
            t_gpu_total += time.time() - t_gpu_start
            t_last_batch_end = time.time()

        n_batches = len(train_dl)
        # Average only over batches that produced a finite loss, so the
        # readout stays meaningful when a few batches are skipped.
        denom = max(n_train_batches, 1)
        train_loss /= denom
        for k in train_metrics:
            train_metrics[k] /= denom
        if nan_batches:
            print(f"  [warn] skipped {nan_batches}/{n_batches} non-finite "
                  f"loss batch(es) this epoch")
            writer.add_scalar("train/nan_batches", nan_batches, ep)

        # Throughput diagnostic. If data_wait_pct > 30%, the dataloader
        # is the bottleneck. If < 10%, the GPU is fully utilized.
        epoch_wall = time.time() - t0
        data_wait_pct = 100 * t_data_wait_total / max(epoch_wall, 0.001)
        gpu_pct = 100 * t_gpu_total / max(epoch_wall, 0.001)
        print(f"  [timing] epoch={epoch_wall:.1f}s, "
              f"data_wait={t_data_wait_total:.1f}s ({data_wait_pct:.0f}%), "
              f"gpu={t_gpu_total:.1f}s ({gpu_pct:.0f}%), "
              f"per-batch={epoch_wall/max(n_batches,1)*1000:.0f}ms")
        writer.add_scalar("perf/epoch_seconds", epoch_wall, ep)
        writer.add_scalar("perf/data_wait_pct", data_wait_pct, ep)
        writer.add_scalar("perf/gpu_pct", gpu_pct, ep)

        # Validation with EMA (skip if not a validation epoch)
        run_val = (ep % args.val_every == 0) or (ep == args.epochs)
        if run_val:
            ema.apply(model)
            model.eval()
            val_loss = 0
            val_metrics = {'dice': 0, 'precision': 0, 'recall': 0, 'fg_pred': 0, 'fg_target': 0}
            n_val = 0
            nan_val_batches = 0
            with torch.inference_mode():
                for img, mask_tgt, wmap, noise, _dist_tgt in tqdm(val_dl, desc="Val"):
                    img = img.to(device, non_blocking=True)
                    mask_tgt = mask_tgt.to(device, non_blocking=True)
                    wmap = wmap.to(device, non_blocking=True)
                    noise = noise.to(device, non_blocking=True)
                    with autocast(device_type='cuda', enabled=args.amp):
                        mask_pred, _, _ = model(img, noise)
                        loss, _ = criterion(mask_pred, mask_tgt, weight_map=wmap)
                    if not torch.isfinite(loss):
                        nan_val_batches += 1
                        continue
                    val_loss += loss.item()
                    m = compute_metrics(mask_pred, mask_tgt)
                    for k in val_metrics:
                        val_metrics[k] += m[k]
                    n_val += 1

            if nan_val_batches:
                print(f"  [warn] skipped {nan_val_batches}/{len(val_dl)} "
                      f"non-finite val loss batch(es)")
                writer.add_scalar("val/nan_batches", nan_val_batches, ep)
            val_loss /= max(n_val, 1)
            for k in val_metrics:
                val_metrics[k] /= max(n_val, 1)
            ema.restore(model)

        # Full-volume sliding-window validation (true stitched Dice/IoU +
        # per-NPC detection F1). Expensive, so gated to every N epochs.
        run_full_val = (args.full_val_every > 0
                        and (ep % args.full_val_every == 0 or ep == args.epochs))
        if run_full_val:
            # Diagnostic only: never let it kill a long training run.
            try:
                ema.apply(model)
                model.eval()
                fv = evaluate_full_volume(
                    model, val_ds, device, args.amp, args.patch_size,
                    args.full_val_stride or args.patch_size // 2,
                    args.full_val_roi, args.full_val_max_volumes, tta=args.tta,
                    npc_group_radius=args.npc_group_radius)
                ema.restore(model)
                print(f"  [full-val] vol_dice={fv['vol_dice']:.4f} "
                      f"vol_iou={fv['vol_iou']:.4f} det_f1={fv['det_f1']:.4f} "
                      f"(P={fv['det_precision']:.3f} R={fv['det_recall']:.3f})")
                for k, v in fv.items():
                    writer.add_scalar(f"fullval/{k}", v, ep)
            except Exception as e:
                print(f"  [full-val] skipped ({type(e).__name__}: {e})")
                ema.restore(model)

        writer.add_scalar("epoch/train_loss", train_loss, ep)
        writer.add_scalar("epoch/lr", opt.param_groups[0]['lr'], ep)

        mode_str = 'FOCAL' if using_focal else 'Tversky'

        if run_val:
            writer.add_scalar("epoch/val_loss", val_loss, ep)
            writer.add_scalar("epoch/val_dice", val_metrics['dice'], ep)
            writer.add_scalar("epoch/val_precision", val_metrics['precision'], ep)
            writer.add_scalar("epoch/val_recall", val_metrics['recall'], ep)
            print(f"\nEpoch {ep}: loss={train_loss:.4f} val={val_loss:.4f} "
                  f"dice={val_metrics['dice']:.4f} prec={val_metrics['precision']:.4f} "
                  f"rec={val_metrics['recall']:.4f} "
                  f"time={time.time()-t0:.0f}s [{mode_str}]")

            # Collapse detection
            if not using_focal and not focal_recovery_done:
                if val_metrics['fg_pred'] < 0.001 and val_metrics['fg_target'] > 0.005:
                    print("  WARNING: Collapse -> Focal+Dice")
                    criterion = loss_focal
                    using_focal = True
            if using_focal and not focal_recovery_done:
                if val_metrics['fg_pred'] > 0.003 and val_metrics['dice'] > 0.1:
                    print("  Recovered -> main loss")
                    criterion = loss_main
                    using_focal = False
                    focal_recovery_done = True
        else:
            print(f"\nEpoch {ep}: loss={train_loss:.4f} "
                  f"time={time.time()-t0:.0f}s [{mode_str}] (val skipped)")

        scheduler.step()

        # Always save latest checkpoint
        ckpt = {
            "epoch": ep, "model": model.state_dict(),
            "opt": opt.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.shadow, "global_step": global_step,
            "patch_size": args.patch_size,
            "focal_recovery_done": focal_recovery_done,
            "val_loss": best_val_loss,
            "val_dice": best_val_dice,
            "val_recall": best_val_recall,
        }
        torch.save(ckpt, args.out)

        # Best-metric checkpoints only when validation was run
        if run_val:
            ckpt["val_loss"] = val_loss
            ckpt["val_dice"] = val_metrics['dice']
            ckpt["val_recall"] = val_metrics['recall']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, args.out.replace(".pt", "_best_loss.pt"))
                print(f"  * Best loss: {val_loss:.4f}")
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                torch.save(ckpt, args.out.replace(".pt", "_best_dice.pt"))
                print(f"  * Best dice: {val_metrics['dice']:.4f}")
            if val_metrics['recall'] > best_val_recall:
                best_val_recall = val_metrics['recall']
                torch.save(ckpt, args.out.replace(".pt", "_best_recall.pt"))
                print(f"  * Best recall: {val_metrics['recall']:.4f}")

    writer.close()
    print(f"\nTRAINING COMPLETE")
    print(f"Best val loss:   {best_val_loss:.4f}")
    print(f"Best val Dice:   {best_val_dice:.4f}")
    print(f"Best val Recall: {best_val_recall:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NPC 3D U-Net Training (v8)")

    # ---- Data: pre-partitioned OR paired flat folder ----
    ap.add_argument("--train_dir", default=None)
    ap.add_argument("--val_dir", default=None)
    ap.add_argument("--paired-dir", default=None,
                    help="Single flat folder of paired image+mask files. "
                         "Requires --paired-image-suffix and "
                         "--paired-mask-suffix.")
    ap.add_argument("--paired-image-suffix", default=None,
                    help="Substring identifying image files "
                         "(e.g. 'pan_processed').")
    ap.add_argument("--paired-mask-suffix", default=None,
                    help="Substring identifying mask files "
                         "(e.g. 'antibody_mask'). Substituted into the "
                         "image suffix to find each pair's mask.")
    ap.add_argument("--partition-root", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.15)

    # ---- Output ----
    ap.add_argument("--out", default="unet3d_npc_v8.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--logdir", default="runs/unet3d_npc_v8")

    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accumulation-steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Initial LR. Cosine-annealed to --min-lr over "
                         "--epochs.")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="Floor for cosine LR annealing.")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--extraction-workers", type=int, default=8,
                    help="Parallel workers for the one-time patch "
                         "extraction pass (cached to disk afterward).")
    ap.add_argument("--stage-to-local", default=None,
                    help="Copy data to a local dir before training. "
                         "Pass an explicit path or 'auto' "
                         "(<paired-dir>/_local_stage). Pre-flight free-"
                         "space check; already-present files skipped.")

    # ---- Model ----
    ap.add_argument("--base", type=int, default=36)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--deep-supervision", action="store_true", default=True,
                    help="Deep supervision at decoder levels (default on).")
    ap.add_argument("--no-deep-supervision", action="store_false",
                    dest="deep_supervision")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aaspp", action="store_true", default=True,
                    help="Replace the U-Net bottleneck with a 3D Atrous Spatial "
                         "Pyramid Pooling module (dilations 1/2/4 + global avg "
                         "pool). Widens receptive field at the bottleneck without "
                         "adding depth. Requires base*2^depth divisible by 4.")
    ap.add_argument("--no-aaspp", action="store_false", dest="aaspp")
    ap.add_argument("--clahe", action="store_true", default=False,
                    help="Weak-signal CLAHE path: white top-hat -> per-slice "
                         "CLAHE -> local-variance soft gate -> MAD norm. "
                         "Enhances dim NPC ring contrast without amplifying "
                         "flat background. Adds CPU cost per patch; monitor "
                         "data_wait_pct.")
    ap.add_argument("--clahe-tophat-size", type=int, default=17,
                    help="White top-hat structuring-element size (~2x NPC "
                         "radius in voxels) for the CLAHE path.")
    ap.add_argument("--clahe-clip-limit", type=float, default=0.01,
                    help="equalize_adapthist clip_limit for the CLAHE path.")

    # ---- Noise awareness / robustness ----
    ap.add_argument("--noise-conditioning", action="store_true", default=True,
                    help="FiLM-condition the network on a per-volume noise "
                         "level so it can adapt to clean vs noisy imaging "
                         "dates instead of averaging over both (default on).")
    ap.add_argument("--no-noise-conditioning", action="store_false",
                    dest="noise_conditioning")
    ap.add_argument("--noise-aug", action="store_true", default=True,
                    help="Noise-matched augmentation: raise cleaner patches "
                         "toward the noisiest tier and set the conditioning "
                         "scalar to match (default on).")
    ap.add_argument("--no-noise-aug", action="store_false", dest="noise_aug")
    ap.add_argument("--noise-aug-gain", type=float, default=0.2,
                    help="Added-noise std (z-scored units) per unit of "
                         "conditioning delta for noise-matched augmentation.")
    ap.add_argument("--bias-field", action="store_true", default=True,
                    help="Smooth multiplicative bias-field augmentation "
                         "(microscopy illumination/staining inhomogeneity; "
                         "default on).")
    ap.add_argument("--no-bias-field", action="store_false",
                    dest="bias_field")
    ap.add_argument("--bias-field-max", type=float, default=0.3,
                    help="Max log-amplitude of the bias field control grid.")
    ap.add_argument("--tta", action="store_true", default=False,
                    help="8-way axis-flip test-time augmentation during "
                         "full-volume validation.")

    # ---- Patches ----
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--max-patches-per-volume", type=int, default=None)
    ap.add_argument("--min-npc-size", type=int, default=1000,
                    help="Minimum connected-component size (voxels) for an "
                         "NPC to be used. Components below this are zeroed "
                         "from the mask at load time (per-patch label_3d "
                         "filter) so the model never sees small specks as "
                         "positives. Default 1000.")
    ap.add_argument("--add-negative-samples", action="store_true")
    ap.add_argument("--negative-ratio", type=float, default=0.3)
    ap.add_argument("--jitter-range", type=int, default=48)
    ap.add_argument("--patches-per-epoch", type=int, default=None,
                    help="Randomly sample this many patches per epoch.")
    ap.add_argument("--fg-ratio", type=float, default=0.7,
                    help="Fraction of each epoch's patches drawn from the "
                         "foreground (NPC-centered) pool; remainder from the "
                         "background/negative pool. Requires --patches-per-epoch. "
                         "Set 0 to fall back to uniform random sampling.")
    ap.add_argument("--val-patches", type=int, default=None,
                    help="Subsample validation to this many patches.")
    ap.add_argument("--val-every", type=int, default=1,
                    help="Run validation every N epochs.")
    ap.add_argument("--full-val-every", type=int, default=0,
                    help="Run sliding-window full-volume validation (stitched "
                         "Dice/IoU + per-NPC detection F1) every N epochs. "
                         "0 disables. Expensive; try 10.")
    ap.add_argument("--full-val-max-volumes", type=int, default=2,
                    help="Number of validation volumes to evaluate per "
                         "full-volume validation pass.")
    ap.add_argument("--full-val-roi", type=int, default=512,
                    help="Edge length of the ROI subvolume evaluated per "
                         "volume during full-volume validation. v9 default "
                         "512 (was 256); 256^3 was <0.5%% of the volume.")
    ap.add_argument("--full-val-stride", type=int, default=0,
                    help="Sliding-window stride for full-volume validation. "
                         "0 -> patch_size // 2.")

    # ---- Loss (scheduled) ----
    # alpha/beta/w_bce/w_cldice/pos_weight accept either a constant ("0.5")
    # or a linear ramp "start:end:epochs" (e.g. "0.3:0.5:40" interpolates
    # from 0.3 to 0.5 over the first 40 training epochs, then holds).
    ap.add_argument("--tversky-alpha", default="0.5:0.7:80",
                    help="Tversky FP weight. Constant or 'start:end:epochs'. "
                         "Default ramps 0.5->0.7 over 80ep: start symmetric, "
                         "then lean precision as label cleaning takes effect "
                         "and clDice tightens predictions.")
    ap.add_argument("--tversky-beta", default="0.5:0.3:80",
                    help="Tversky FN weight. Constant or 'start:end:epochs'. "
                         "Default ramps 0.5->0.3 over 80ep: mirror of alpha "
                         "ramp, reducing FN penalty as precision improves.")
    ap.add_argument("--w-bce", default="0.3",
                    help="SCE term weight. Constant or 'start:end:epochs'.")
    ap.add_argument("--w-cldice", default="0:0.7:30",
                    help="soft-clDice term weight. Constant or "
                         "'start:end:epochs'. Default warms 0->0.7 over 30 ep "
                         "(skeletons of early predictions are meaningless; "
                         "higher ceiling + slower ramp enforces tighter ring/"
                         "disk topology and reduces diffuse false positives).")
    ap.add_argument("--pos-weight", default="1.5",
                    help="CE positive-class weight. Constant or "
                         "'start:end:epochs'. v8 default 1.5 (was 5): with "
                         "trustworthy negatives but noisy positives, a large "
                         "pos_weight amplifies every spurious positive label. "
                         "Keep it mild -- just enough to learn the rare class.")
    ap.add_argument("--w-gce", type=float, default=0.5,
                    help="Weight of the GCE/Lq robust-positive term (Zhang & "
                         "Sabuncu 2018) on foreground voxels. Bounded-gradient "
                         "loss so spurious positive labels cannot dominate. "
                         "0 disables.")
    ap.add_argument("--gce-q", type=float, default=0.7,
                    help="GCE/Lq exponent q in (0,1]. q->0 approaches CE; "
                         "q=1 is the noise-robust MAE. 0.7 is a robust default.")
    ap.add_argument("--w-dist", type=float, default=0.1,
                    help="Weight for the distance heatmap MSE auxiliary loss "
                         "(v9). Supervises the dist output head with a "
                         "Gaussian-blurred binary mask so the head provides "
                         "a topology-agnostic NPC centroid heatmap. 0 disables.")
    ap.add_argument("--dist-sigma", type=float, default=4.0,
                    help="Gaussian sigma (voxels) for the distance heatmap "
                         "target. Controls how broadly the heatmap spreads "
                         "around foreground voxels.")
    ap.add_argument("--w-dice", type=float, default=1.0,
                    help="Focal-Tversky term weight (constant).")
    ap.add_argument("--ft-gamma", type=float, default=1.333,
                    help="Focal-Tversky exponent (1-TI)**gamma; >1 focuses "
                         "on hard patches.")
    ap.add_argument("--sce-alpha", type=float, default=1.0,
                    help="SCE weight on forward (pos-weighted) CE.")
    ap.add_argument("--sce-beta", type=float, default=0.5,
                    help="SCE weight on reverse CE (RCE); raise toward 1.0 "
                         "if mask label noise exceeds ~5%%.")
    ap.add_argument("--cldice-iters", type=int, default=5,
                    help="Soft-skeletonization iterations for clDice.")
    ap.add_argument("--z-scale-range", type=float, nargs=2,
                    default=[0.5, 2.0], metavar=("LO", "HI"),
                    help="Independent Z-axis scale range for anisotropic "
                         "augmentation. Compressing Z (scale_z > XY scale) "
                         "simulates face-on ring morphology; stretching Z "
                         "simulates side-on 3-dot morphology. Applied every "
                         "time the patch is in affine mode (SO(3)/cube-affine).")

    # ---- Performance ----
    ap.add_argument("--lr-warmup-epochs", type=int, default=5,
                    help="Number of epochs for linear LR warmup before "
                         "cosine annealing begins (v9). Stabilises early "
                         "training with the 6-term compound loss. 0 disables.")
    ap.add_argument("--npc-group-radius", type=int, default=20,
                    help="Dilation radius (voxels) for NPC-level detection "
                         "grouping in full-volume validation (v9). Dilates "
                         "both pred and GT before component matching so "
                         "co-NPC rings merge into one instance. 0 disables "
                         "(reverts to v8 ring-level matching).")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--preload-volumes", action="store_true", default=False,
                    help="Load all image and mask volumes into OS shared memory "
                         "at startup. Workers then access volumes via zero-copy "
                         "tensor slices instead of per-getitem TIF page reads. "
                         "Requires ~1 TB RAM for a 125-volume dataset at "
                         "400x2720x2720. One-time load cost (~2-5 min) pays off "
                         "across all training epochs.")

    args = ap.parse_args()

    # ---- Validate ----
    if args.paired_dir:
        if (args.paired_image_suffix is None
                or args.paired_mask_suffix is None):
            ap.error("--paired-dir requires both --paired-image-suffix "
                     "and --paired-mask-suffix")
    if args.train_dir and not args.val_dir:
        ap.error("--train_dir requires --val_dir")
    if args.val_dir and not args.train_dir:
        ap.error("--val_dir requires --train_dir")
    if not args.train_dir and not args.paired_dir:
        ap.error("Provide --train_dir + --val_dir, or --paired-dir "
                 "+ --paired-image-suffix + --paired-mask-suffix")

    # Validate schedule specs early so a typo fails before the long
    # extraction step rather than after.
    for name in ("tversky_alpha", "tversky_beta", "w_bce", "w_cldice",
                 "pos_weight"):
        try:
            parse_schedule(getattr(args, name))
        except Exception as e:
            ap.error(f"--{name.replace('_','-')}: {e}")
    train(args)
