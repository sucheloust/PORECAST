"""Patch and full-volume validation metrics: Dice/precision/recall, Gaussian-
blended sliding-window inference, flip TTA, and per-NPC detection F1."""

import numpy as np
import tifffile
import torch
from torch.amp import autocast
from scipy.ndimage import label as label_3d, binary_dilation


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
    matching, so a multi-ring NPC counts as a single object. GT components
    below min_size are ignored; predicted components are all counted.
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
