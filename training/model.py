"""3D U-Net building blocks, the network itself, and the EMA wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    3D Atrous Spatial Pyramid Pooling bottleneck. Three parallel dilated-conv
    branches (rates 1, 2, 4) plus a global avg-pool branch, concatenated and
    fused with a 1x1 conv and a residual shortcut. Widens the receptive field
    without adding depth. Optional FiLM conditioning is applied after the fuse.
    Requires out_c divisible by 4.
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
