#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined pan-stain + antibody-stain ND2 batch processor with z-anisotropy correction.

Per ND2 file, produces:
    {stem}_pan_processed_v9.tif       (uint16, contrast-stretched, z-resampled)
    {stem}_antibody_mask_v14.tif      (uint8 binary mask, z-resampled, pan-boundary snapped)

Pan processing:  percentile stretch (nd2_to_tif.py lineage).
Antibody mask:   triangle gate + soft pan-boundary smoothing (mask_gen_v2.py lineage).
Anisotropy fix:  Z axis resampled by --z-aniso (default 2.215) immediately after ND2 load,
                 so output voxels are isotropic (1 px = 1 physical unit in all axes).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile
from nd2reader import ND2Reader
from scipy.ndimage import zoom as ndimage_zoom

from skimage.filters import gaussian, median, threshold_triangle
from skimage.morphology import white_tophat, disk
from skimage import exposure, measure
from scipy.ndimage import label as cpu_label_3d
from joblib import Parallel, delayed

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        gaussian_filter as cp_gaussian_filter,
        median_filter as cp_median_filter,
        grey_opening as cp_grey_opening,
    )
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


# ==============================================================
# MODULE-LEVEL GLOBALS (set by main / worker init)
# ==============================================================

INPUT_ROOT: Path = None   # type: ignore[assignment]
OUTPUT_ROOT: Path = None  # type: ignore[assignment]

PAN_CHANNEL = 1
ANTIBODY_CHANNEL = 0
Z_ANISO = 2.215           # physical z-spacing / xy-spacing; applied as zoom factor on z

# ----- Pan stretch -----
PAN_LOW_PCTL = 1.0
PAN_HIGH_PCTL = 99.5
PAN_OUTPUT_DTYPE = np.uint16
PAN_OUT_SUFFIX = "_pan_processed_v9.tif"

# ----- Antibody preprocessing -----
TOPHAT_RADIUS = 15
ANTIBODY_VOL_PCTL_LOW  = 1.0
ANTIBODY_VOL_PCTL_HIGH = 99.5
SLICE_NORM_PCTL_LOW = 2
SLICE_NORM_PCTL_HIGH = 98
GAUSSIAN_SIGMA = 1

# ----- Voxel threshold -----
VOXEL_THRESHOLD_SCALE = 1.75   # triangle threshold multiplier (mask_gen_v2 value)

# ----- Object size gates -----
MIN_OBJECT_VOXELS   = 1000
MIN_OBJECT_Z_EXTENT = 3

# ----- Pan-boundary smoothing -----
PAN_SMOOTH_SIGMA   = 2.0
PAN_BOUNDARY_SIGMA = 1.5
PAN_BG_PCTILE      = 10.0
PAN_DELTA          = 0.20
MASK_OUT_SUFFIX    = "_antibody_mask_v14.tif"

# ----- Resource control -----
N_JOBS_PER_FILE = max(1, (os.cpu_count() or 4) - 1)
HIST_BINS = 4096
HIST_PERCENTILE_SAMPLE_CAP = 50_000_000

DEVICE = None  # set in main()


# ==============================================================
# DEVICE SELECTION
# ==============================================================

def select_device() -> str:
    if _HAS_CUPY:
        try:
            cp.cuda.runtime.getDeviceCount()
            return "cupy"
        except Exception:
            pass
    if _HAS_TORCH and torch.cuda.is_available():
        return "torch_cuda"
    return "cpu"


def _free_gpu_memory():
    if _HAS_CUPY:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


# ==============================================================
# FAST HISTOGRAM PERCENTILES
# ==============================================================

def histogram_percentiles(
    data: np.ndarray,
    low_pct: float,
    high_pct: float,
    bins: int = HIST_BINS,
) -> Tuple[float, float]:
    flat = data.ravel()
    if flat.size > HIST_PERCENTILE_SAMPLE_CAP:
        step = flat.size // HIST_PERCENTILE_SAMPLE_CAP + 1
        flat = flat[::step]
    if flat.size == 0:
        raise ValueError("Empty data for percentile computation.")
    dmin, dmax = float(flat.min()), float(flat.max())
    if dmin == dmax:
        return dmin, dmax
    hist, edges = np.histogram(flat, bins=bins, range=(dmin, dmax))
    cdf = np.cumsum(hist)
    total = cdf[-1]
    if total == 0:
        return dmin, dmax
    lo_idx = min(max(int(np.searchsorted(cdf, total * low_pct / 100.0)), 0), len(edges) - 2)
    hi_idx = min(max(int(np.searchsorted(cdf, total * high_pct / 100.0)), 0), len(edges) - 2)
    return float(edges[lo_idx]), float(edges[hi_idx + 1])


# ==============================================================
# ND2 LOADING + Z-RESAMPLING
# ==============================================================

def load_nd2_channel(nd2_path: Path, channel: int) -> Tuple[np.ndarray, int]:
    """Load one channel as (Z, Y, X) float32. Returns (volume, n_channels)."""
    reader = ND2Reader(str(nd2_path))
    try:
        size_z = reader.sizes.get("z", 1)
        size_c = reader.sizes.get("c", 1)
        size_y = reader.sizes["y"]
        size_x = reader.sizes["x"]
        if channel >= size_c:
            raise RuntimeError(f"Channel {channel} requested, file has {size_c} channels")
        out = np.empty((size_z, size_y, size_x), dtype=np.float32)
        if size_c == 1:
            reader.bundle_axes = "yx"
            reader.iter_axes = "z"
            for z, frame in enumerate(reader):
                out[z] = np.asarray(frame, dtype=np.float32)
        else:
            reader.bundle_axes = "cyx"
            reader.iter_axes = "z"
            for z, frame in enumerate(reader):
                out[z] = np.asarray(frame[channel], dtype=np.float32)
        return out, size_c
    finally:
        reader.close()


def resample_z(vol: np.ndarray, z_factor: float) -> np.ndarray:
    """Upsample Z axis to correct physical anisotropy. order=1 (linear)."""
    if abs(z_factor - 1.0) < 1e-6:
        return vol
    return ndimage_zoom(vol, (z_factor, 1.0, 1.0), order=1).astype(vol.dtype)


# ==============================================================
# PAN STAIN: CONTRAST STRETCH (from nd2_to_tif.py)
# ==============================================================

def process_pan_stain(stack: np.ndarray, out_dtype=PAN_OUTPUT_DTYPE,
                      z_chunk: int = 64) -> np.ndarray:
    lo, hi = histogram_percentiles(stack, PAN_LOW_PCTL, PAN_HIGH_PCTL)
    if hi <= lo:
        return np.zeros_like(stack, dtype=out_dtype)
    info = np.iinfo(out_dtype)
    scale = info.max / (hi - lo)

    if DEVICE == "cupy" and _HAS_CUPY:
        Z = stack.shape[0]
        out = np.empty(stack.shape, dtype=out_dtype)
        for z0 in range(0, Z, z_chunk):
            z1 = min(Z, z0 + z_chunk)
            t = cp.asarray(stack[z0:z1], dtype=cp.float32)
            cp.clip(t, lo, hi, out=t)
            t -= lo
            t *= scale
            cp.clip(t, info.min, info.max, out=t)
            out[z0:z1] = cp.asnumpy(t).astype(out_dtype)
            del t
            cp.get_default_memory_pool().free_all_blocks()
        return out

    if DEVICE == "torch_cuda" and _HAS_TORCH:
        Z = stack.shape[0]
        out = np.empty(stack.shape, dtype=out_dtype)
        for z0 in range(0, Z, z_chunk):
            z1 = min(Z, z0 + z_chunk)
            t = torch.from_numpy(stack[z0:z1]).to("cuda", dtype=torch.float32)
            t = torch.clamp(t, lo, hi)
            t.sub_(lo).mul_(scale).clamp_(info.min, info.max)
            out[z0:z1] = t.cpu().numpy().astype(out_dtype)
            del t
        torch.cuda.empty_cache()
        return out

    clipped = np.clip(stack, lo, hi).astype(np.float32)
    clipped -= lo
    clipped *= scale
    return np.clip(clipped, info.min, info.max).astype(out_dtype)


# ==============================================================
# ANTIBODY PREPROCESSING (from mask_gen_v2.py / nd2_to_tif.py — identical)
# ==============================================================

def _preprocess_antibody_slice(
    img: np.ndarray,
    sigma: float,
    tophat_radius: int,
    clip_lo: Optional[float] = None,
    clip_hi: Optional[float] = None,
) -> np.ndarray:
    if clip_lo is not None and clip_hi is not None and clip_hi > clip_lo:
        p_low, p_high = clip_lo, clip_hi
    else:
        p_low, p_high = np.percentile(img, (SLICE_NORM_PCTL_LOW, SLICE_NORM_PCTL_HIGH))
    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.float32)
    img_norm = exposure.rescale_intensity(img, in_range=(p_low, p_high))
    smoothed = gaussian(img_norm, sigma=sigma, preserve_range=True)
    med_filtered = median(smoothed, footprint=disk(1))
    tophat = white_tophat(med_filtered, footprint=disk(tophat_radius))
    return np.log1p(tophat).astype(np.float32)


def _disk_footprint_gpu(radius: int):
    r = int(radius)
    yy, xx = cp.ogrid[-r:r + 1, -r:r + 1]
    disk2d = (yy * yy + xx * xx) <= r * r
    return disk2d[None, :, :]


def _preprocess_antibody_gpu(
    vol: np.ndarray,
    global_lo: float,
    global_hi: float,
    z_chunk: int = 32,
) -> np.ndarray:
    global_rng = max(global_hi - global_lo, 1e-6)
    Z = vol.shape[0]
    out = np.empty_like(vol, dtype=np.float32)
    disk3 = _disk_footprint_gpu(TOPHAT_RADIUS)
    for z0 in range(0, Z, z_chunk):
        z1 = min(Z, z0 + z_chunk)
        g = cp.asarray(vol[z0:z1], dtype=cp.float32)
        cp.clip(g, global_lo, global_hi, out=g)
        g = (g - global_lo) / global_rng
        g = cp_gaussian_filter(g, sigma=(0, GAUSSIAN_SIGMA, GAUSSIAN_SIGMA))
        g = cp_median_filter(g, size=(1, 3, 3))
        opened = cp_grey_opening(g, footprint=disk3)
        g = g - opened
        del opened
        cp.clip(g, 0, None, out=g)
        g = cp.log1p(g).astype(cp.float32)
        out[z0:z1] = cp.asnumpy(g)
        del g
        cp.get_default_memory_pool().free_all_blocks()
    return out


def preprocess_antibody(vol: np.ndarray, n_jobs: int = N_JOBS_PER_FILE) -> np.ndarray:
    global_lo, global_hi = histogram_percentiles(
        vol, ANTIBODY_VOL_PCTL_LOW, ANTIBODY_VOL_PCTL_HIGH)
    if DEVICE == "cupy" and _HAS_CUPY:
        return _preprocess_antibody_gpu(vol, global_lo, global_hi)
    processed = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_preprocess_antibody_slice)(
            img, GAUSSIAN_SIGMA, TOPHAT_RADIUS, clip_lo=global_lo, clip_hi=global_hi,
        )
        for img in vol
    )
    return np.stack(processed)


# ==============================================================
# ANTIBODY MASK (from mask_gen_v2.py — triangle gate, size/extent gates only)
# ==============================================================

def generate_antibody_mask(processed_vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    positive = processed_vol[processed_vol > 0]
    if positive.size == 0:
        empty = np.zeros_like(processed_vol, dtype=bool)
        return empty, np.zeros_like(processed_vol, dtype=np.int32)

    voxel_cut = float(threshold_triangle(positive)) * VOXEL_THRESHOLD_SCALE
    print(f"  voxel_cut={voxel_cut:.4f} (triangle * {VOXEL_THRESHOLD_SCALE})", flush=True)

    voxel_mask = processed_vol > voxel_cut
    labeled = measure.label(voxel_mask, connectivity=3)
    props = measure.regionprops(labeled, intensity_image=processed_vol)

    if not props:
        return voxel_mask, labeled

    keep_labels = []
    for p in props:
        z0, _, _, z1, _, _ = p.bbox
        if p.area < MIN_OBJECT_VOXELS:
            continue
        if (z1 - z0) < MIN_OBJECT_Z_EXTENT:
            continue
        keep_labels.append(p.label)

    keep_arr = np.isin(labeled, keep_labels)
    relabeled, _ = cpu_label_3d(keep_arr)
    return keep_arr, relabeled.astype(np.int32)


# ==============================================================
# PAN-BOUNDARY SMOOTHING (from mask_gen_v2.py)
# ==============================================================

def smooth_mask_to_pan_boundaries(mask: np.ndarray, pan_path: Path) -> np.ndarray:
    if pan_path is None or not pan_path.exists():
        print(f"  pan file not found, skipping boundary smoothing: "
              f"{pan_path.name if pan_path else 'N/A'}", flush=True)
        return mask

    try:
        pan = tifffile.imread(str(pan_path)).astype(np.float32)
    except Exception as e:
        print(f"  could not load pan file ({e}), skipping boundary smoothing", flush=True)
        return mask

    while pan.ndim > 3 and pan.shape[0] == 1:
        pan = pan[0]
    if pan.ndim > 3:
        pan = pan[0]

    if pan.shape != mask.shape:
        print(f"  pan shape {pan.shape} != mask shape {mask.shape}, "
              f"skipping boundary smoothing", flush=True)
        return mask

    pan_sm = gaussian(pan, sigma=PAN_SMOOTH_SIGMA, preserve_range=True)
    p_lo = float(np.percentile(pan_sm, PAN_BG_PCTILE))
    p_hi = float(np.percentile(pan_sm, 99.5))
    if p_hi <= p_lo:
        return mask
    pan_norm = np.clip((pan_sm - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)
    del pan, pan_sm

    mask_soft = gaussian(mask.astype(np.float32), sigma=PAN_BOUNDARY_SIGMA,
                         preserve_range=True).astype(np.float32)
    is_boundary = (mask_soft > 0.05) & (mask_soft < 0.95)
    adaptive_thresh = (0.5 - PAN_DELTA * (pan_norm - 0.5)).astype(np.float32)
    boundary_decision = mask_soft > adaptive_thresh

    refined = np.where(is_boundary, boundary_decision, mask.astype(bool))
    n_added = int(refined.sum()) - int(mask.sum())
    print(f"  pan boundary smoothing: boundary voxels delta={n_added:+d}", flush=True)
    return refined.astype(bool)


# ==============================================================
# PER-FILE PIPELINE
# ==============================================================

def process_one(nd2_path: Path, index: Optional[int] = None,
                total: Optional[int] = None) -> str:
    nd2_path = Path(nd2_path)
    stem = nd2_path.stem
    try:
        rel = nd2_path.relative_to(INPUT_ROOT)
    except ValueError:
        rel = nd2_path
    outdir = OUTPUT_ROOT / nd2_path.parent.relative_to(INPUT_ROOT)
    outdir.mkdir(parents=True, exist_ok=True)

    pan_out  = outdir / f"{stem}{PAN_OUT_SUFFIX}"
    mask_out = outdir / f"{stem}{MASK_OUT_SUFFIX}"

    progress = f"[{index}/{total}] " if (index is not None and total is not None) else ""

    if pan_out.exists() and mask_out.exists():
        print(f"{progress}SKIP  {rel}  (outputs exist)", flush=True)
        return f"✔ skip {nd2_path.name}"

    t0 = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] {progress}START {rel}  "
          f"(PID {os.getpid()}, device={DEVICE})", flush=True)

    ab_stack = ab_processed = pan_stack = pan_uint16 = raw_mask = labeled = None

    try:
        # ----------------------------------------------------------
        # Phase 1: ANTIBODY — load, z-resample, preprocess, gate
        # ----------------------------------------------------------
        t = time.time()
        ab_stack, n_channels = load_nd2_channel(nd2_path, ANTIBODY_CHANNEL)
        if PAN_CHANNEL >= n_channels:
            raise RuntimeError(
                f"Pan channel {PAN_CHANNEL} requested, file has {n_channels}")
        print(f"  loaded antibody ch={ANTIBODY_CHANNEL} "
              f"shape={tuple(ab_stack.shape)} in {time.time() - t:.1f}s", flush=True)

        t = time.time()
        ab_stack = resample_z(ab_stack, Z_ANISO)
        print(f"  z-resampled (x{Z_ANISO}) -> {tuple(ab_stack.shape)} "
              f"in {time.time() - t:.1f}s", flush=True)

        if not mask_out.exists():
            t = time.time()
            ab_processed = preprocess_antibody(ab_stack)
            ab_stack = None
            _free_gpu_memory()
            print(f"  antibody preprocess ({time.time() - t:.1f}s)", flush=True)

            t = time.time()
            raw_mask, labeled = generate_antibody_mask(ab_processed)
            ab_processed = None
            n_obj = int(labeled.max()) if labeled.size else 0
            if n_obj < 65535:
                labeled = labeled.astype(np.uint16, copy=False)
            print(f"  mask generation: {n_obj} objects survived gates "
                  f"({time.time() - t:.1f}s)", flush=True)
        else:
            ab_stack = None
            print(f"  mask: cached {mask_out.name}", flush=True)

        # ----------------------------------------------------------
        # Phase 2: PAN — load, z-resample, stretch, save
        # ----------------------------------------------------------
        t = time.time()
        pan_stack, _ = load_nd2_channel(nd2_path, PAN_CHANNEL)
        print(f"  loaded pan ch={PAN_CHANNEL} shape={tuple(pan_stack.shape)} "
              f"in {time.time() - t:.1f}s", flush=True)

        if not pan_out.exists():
            t = time.time()
            pan_stack = resample_z(pan_stack, Z_ANISO)
            pan_uint16 = process_pan_stain(pan_stack)
            pan_stack = None
            tifffile.imwrite(pan_out, pan_uint16, bigtiff=True)
            pan_uint16 = None
            print(f"  pan stretch + save -> {pan_out.name} "
                  f"({time.time() - t:.1f}s)", flush=True)
        else:
            pan_stack = None
            print(f"  pan: cached {pan_out.name}", flush=True)

        # ----------------------------------------------------------
        # Phase 3: MASK — pan-boundary smoothing, save
        # ----------------------------------------------------------
        if not mask_out.exists():
            if raw_mask is not None and raw_mask.any():
                t = time.time()
                raw_mask = smooth_mask_to_pan_boundaries(raw_mask, pan_out)
                print(f"  pan boundary smoothing ({time.time() - t:.1f}s)", flush=True)
            else:
                print("  no objects survived — writing empty mask", flush=True)
                raw_mask = np.zeros(
                    tuple(int(round(s * (Z_ANISO if i == 0 else 1.0)))
                          for i, s in enumerate(labeled.shape if labeled is not None
                                                else (1, 1, 1))),
                    dtype=np.uint8)

            tifffile.imwrite(mask_out, raw_mask.astype(np.uint8), bigtiff=True)
            print(f"  mask saved -> {mask_out.name}", flush=True)
            raw_mask = labeled = None

        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {progress}DONE  {rel}  "
              f"({elapsed:.1f}s)", flush=True)
        return f"✓ {nd2_path.name} ({elapsed:.1f}s)"

    finally:
        ab_stack = ab_processed = pan_stack = pan_uint16 = None
        raw_mask = labeled = None
        _free_gpu_memory()


def process_one_safe(args: Tuple[Path, int, int]) -> str:
    nd2_path, index, total = args
    try:
        return process_one(nd2_path, index=index, total=total)
    except Exception as e:
        msg = f"❌ [{index}/{total}] {Path(nd2_path).name}: {e}"
        print(msg, flush=True)
        print(traceback.format_exc(), flush=True)
        return msg


# ==============================================================
# BATCH ENTRY POINT
# ==============================================================

def main():
    global DEVICE, Z_ANISO

    parser = argparse.ArgumentParser(
        description="ND2 → isotropic TIF preprocessor (pan v9 + antibody mask v14).")
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Root directory containing .nd2 files (searched recursively).")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root directory for output TIFFs.")
    parser.add_argument("--pan-channel", type=int, default=PAN_CHANNEL)
    parser.add_argument("--antibody-channel", type=int, default=ANTIBODY_CHANNEL)
    parser.add_argument("--z-aniso", type=float, default=Z_ANISO,
                        help="Physical z-spacing / xy-spacing. Applied as zoom factor "
                             "to the z axis immediately after loading to produce isotropic "
                             f"output voxels. Default: {Z_ANISO}.")
    parser.add_argument("--file", type=Path, action="append", default=None,
                        help="Process specific ND2 file(s) instead of the full input root.")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU worker processes (0 = auto). Ignored when GPU is used.")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    if args.pan_channel == args.antibody_channel:
        parser.error("--pan-channel and --antibody-channel must differ")

    globals()["INPUT_ROOT"] = args.input_root
    globals()["OUTPUT_ROOT"] = args.output_root
    globals()["PAN_CHANNEL"] = args.pan_channel
    globals()["ANTIBODY_CHANNEL"] = args.antibody_channel
    Z_ANISO = args.z_aniso

    DEVICE = "cpu" if args.force_cpu else select_device()
    print(f"Backend:  {DEVICE}")
    print(f"Input:    {args.input_root}")
    print(f"Output:   {args.output_root}")
    print(f"Channels: pan={args.pan_channel}, antibody={args.antibody_channel}")
    print(f"Z aniso:  {Z_ANISO}x  ({args.input_root} voxels -> isotropic)")

    nd2_files = args.file if args.file else sorted(args.input_root.rglob("*.nd2"))
    total = len(nd2_files)
    print(f"Found {total} ND2 files.")
    if not nd2_files:
        return

    work = [(f, i + 1, total) for i, f in enumerate(nd2_files)]
    batch_t0 = time.time()
    n_ok = n_skip = n_fail = 0

    def _tally(result: str):
        nonlocal n_ok, n_skip, n_fail
        if result.startswith("✓"):
            n_ok += 1
        elif result.startswith("✔"):
            n_skip += 1
        else:
            n_fail += 1

    if DEVICE in ("cupy", "torch_cuda"):
        for item in work:
            _tally(process_one_safe(item))
    else:
        n_proc = args.workers or max(1, min(total, (os.cpu_count() or 4) // 2))
        print(f"Running with {n_proc} CPU processes.")
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_proc,
            initializer=_worker_init,
            initargs=(DEVICE, args.input_root, args.output_root,
                      args.pan_channel, args.antibody_channel, Z_ANISO),
        ) as pool:
            for result in pool.imap_unordered(process_one_safe, work):
                _tally(result)

    elapsed = time.time() - batch_t0
    print(f"\n========== BATCH COMPLETE ==========")
    print(f"  processed: {n_ok}")
    print(f"  skipped:   {n_skip}")
    print(f"  failed:    {n_fail}")
    print(f"  total time: {elapsed / 60:.1f} min "
          f"({elapsed / max(total, 1):.1f}s avg per file)")


def _worker_init(device, input_root, output_root, pan_channel, antibody_channel, z_aniso):
    global DEVICE, INPUT_ROOT, OUTPUT_ROOT, PAN_CHANNEL, ANTIBODY_CHANNEL, Z_ANISO
    DEVICE = device
    INPUT_ROOT = input_root
    OUTPUT_ROOT = output_root
    PAN_CHANNEL = pan_channel
    ANTIBODY_CHANNEL = antibody_channel
    Z_ANISO = z_aniso


if __name__ == "__main__":
    main()
