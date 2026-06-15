"""3D geometric and intensity augmentations: cube/SO(3) rotations, affine
rescale, B-spline elastic deformation, and random bias fields."""

import numpy as np
import torch
import torch.nn.functional as F


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
