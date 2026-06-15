"""Dataset, patch extraction, paired-folder partitioning, local-disk staging,
and the stratified patch sampler."""

import hashlib
import multiprocessing as mp
import os
import pickle
import shutil
import time
from pickle import PicklingError
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, Sampler
from scipy.ndimage import label as label_3d, gaussian_filter
from skimage.measure import regionprops
from tqdm import tqdm

from augment import (CUBE_ROTATIONS, apply_cube_rotation,
                     _random_rotation_matrix_so3, apply_affine_3d,
                     elastic_deform_3d, random_bias_field_3d)
from preprocessing import (NOISE_COND_SCALE, compute_volume_stats_streaming,
                           clahe_weak_signal_filter)


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

    # Symlink into the conventional layout. Preserves the original
    # filenames (so vol_01_img.tif goes to train/images/vol_01_img.tif).
    # The dataset matches images to masks by sorted order within each
    # subdir
    for i, (ip, mk) in enumerate(zip(img_files, mask_files)):
        dest = val_dir if i in val_set else train_dir
        for sub, src in [("images", ip), ("masks", mk)]:
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
        """Apply the transforms chosen in the pre-sampled plan."""
        # SO(3) rotations expose top-down (en face) orientations that the 24
        # axis-permuting cube rotations cannot reach.
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
        # Additive noise is applied by the noise-matched augmentation in
        # __getitem__ (which also updates the conditioning scalar).
        return img.astype(np.float32), mask.astype(np.float32)


def _bytes_human(n):
    """Format a byte count as a human-readable string."""
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
                shutil.copy2(str(real_src), str(tmp_target))
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
