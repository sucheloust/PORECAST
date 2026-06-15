#!/usr/bin/env python3
"""Training entry point: data loading, optimization loop, validation,
checkpointing, and the command-line interface."""

import argparse
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from schedules import parse_schedule
from model import UNet3D, EMAModel
from losses import TverskyDiceLoss, FocalDiceLoss
from data import (NPCDataset3D, StratifiedPatchSampler,
                  partition_train_val_paired, stage_to_local_disk)
from evaluate import compute_metrics, evaluate_full_volume


# ---------------------------------------------------------------------------
# GPU Keepalive: Optional if CUDA times out during initial data load
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
        if not Path(args.resume).exists():
            print(f"\n--resume {args.resume} does not exist; starting fresh.")
        else:
            print(f"\nResuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            # Load only checkpoint tensors whose shape matches the current
            # model, so architecture changes (--base, --depth, --patch-size)
            # don't break resume. strict=False tolerates missing/extra keys
            # but not shape mismatches, so they are filtered explicitly.
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

                # Deep supervision loss: weighted sum of BCE at each decoder
                # level, with exponentially decaying weights for coarser
                # levels. pos_weight keeps coarse heads from collapsing to
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
                # and the Gaussian-blurred binary target (topology-agnostic
                # NPC centroid heatmap).
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
    ap = argparse.ArgumentParser(description="NPC 3D U-Net Training")

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
    ap.add_argument("--out", default="unet3d_npc.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--logdir", default="runs/unet3d_npc")

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
                         "volume during full-volume validation.")
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
                         "'start:end:epochs'. Kept mild: with trustworthy "
                         "negatives but noisy positives, a large pos_weight "
                         "amplifies every spurious positive label.")
    ap.add_argument("--w-gce", type=float, default=0.5,
                    help="Weight of the GCE/Lq robust-positive term (Zhang & "
                         "Sabuncu 2018) on foreground voxels. Bounded-gradient "
                         "loss so spurious positive labels cannot dominate. "
                         "0 disables.")
    ap.add_argument("--gce-q", type=float, default=0.7,
                    help="GCE/Lq exponent q in (0,1]. q->0 approaches CE; "
                         "q=1 is the noise-robust MAE. 0.7 is a robust default.")
    ap.add_argument("--w-dist", type=float, default=0.1,
                    help="Weight for the distance heatmap MSE auxiliary loss. "
                         "Supervises the dist output head with a Gaussian-"
                         "blurred binary mask so the head provides a topology-"
                         "agnostic NPC centroid heatmap. 0 disables.")
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
                         "cosine annealing begins. Stabilises early training "
                         "with the compound loss. 0 disables.")
    ap.add_argument("--npc-group-radius", type=int, default=20,
                    help="Dilation radius (voxels) for NPC-level detection "
                         "grouping in full-volume validation. Dilates both "
                         "pred and GT before component matching so co-NPC "
                         "rings merge into one instance. 0 disables "
                         "(ring-level matching).")
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
