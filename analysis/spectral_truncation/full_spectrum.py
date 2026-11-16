"""FourierConv2d variant that keeps both signs of kx (proposed fix for phd-thesis#85)."""

import torch

from dafx22_fno.modules.layers import FourierConv2d


class FourierConv2dFullSpectrum(FourierConv2d):
    """FourierConv2d with a second weight (and bias) block for the negative-kx rows.

    The zero-padded grid has 2 * size_x rows, so rows [:size_x] (kx >= 0) and
    [-size_x:] (kx < 0) cover every kx; the original layer keeps only the first block.
    """

    def __init__(self, in_channels, out_channels, size_x, size_y, bias=True):
        super().__init__(in_channels, out_channels, size_x, size_y, bias=bias)
        scale = 1 / (in_channels * out_channels)
        self.weights_neg = torch.nn.Parameter(
            torch.view_as_real(
                scale
                * torch.rand(
                    in_channels, out_channels, size_x, size_y, dtype=torch.cfloat
                )
            )
        )
        self.biases_neg = torch.nn.Parameter(
            torch.view_as_real(
                (1 / out_channels)
                * torch.rand(out_channels, size_x, size_y, dtype=torch.cfloat)
            )
        )

    def forward(self, x):
        sx, sy = self.size_x, self.size_y
        x_ft = torch.fft.rfft2(torch.nn.functional.pad(x, [0, sy, 0, sx]))
        out_ft = torch.zeros_like(x_ft)
        blocks = (
            (slice(None, sx), self.weights, self.biases),
            (slice(-sx, None), self.weights_neg, self.biases_neg),
        )
        for rows, weights, biases in blocks:
            out_ft[:, :, rows, :sy] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, rows, :sy],
                torch.view_as_complex(weights),
            )
            if self.bias:
                out_ft[:, :, rows, :sy] += torch.view_as_complex(biases)
        return torch.fft.irfft2(out_ft)[..., :sx, :sy]


def use_full_spectrum(module):
    """Replace every FourierConv2d inside `module` by FourierConv2dFullSpectrum, in place."""
    for name, child in module.named_children():
        if isinstance(child, FourierConv2d) and not isinstance(
            child, FourierConv2dFullSpectrum
        ):
            setattr(
                module,
                name,
                FourierConv2dFullSpectrum(
                    child.in_channels,
                    child.out_channels,
                    child.size_x,
                    child.size_y,
                    bias=child.bias,
                ),
            )
        else:
            use_full_spectrum(child)
    return module
