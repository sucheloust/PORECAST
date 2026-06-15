"""Segmentation losses: soft clDice, focal-Tversky + SCE + GCE compound loss,
and a focal+Dice fallback for collapse recovery."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from schedules import ConstantSchedule


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
    labels (e.g. antibody false positives) cannot dominate the loss the way
    an aggressive pos-weighted BCE term does.
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
