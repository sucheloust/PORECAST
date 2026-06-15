"""Per-volume normalization statistics, Immerkaer noise estimation, and the
CLAHE weak-signal enhancement filter."""

import numpy as np
import tifffile
from scipy.ndimage import convolve, white_tophat, uniform_filter
from skimage.exposure import equalize_adapthist


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
