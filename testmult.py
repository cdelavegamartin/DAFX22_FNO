import torch
from functools import partial
import copy

from dafx22_fno.modules.fno_ref import SpectralConv2d
from dafx22_fno.modules.fno_ref import SpectralConv1d

from dafx22_fno.modules.layers import FourierConv1d
from dafx22_fno.modules.layers import FourierConv2d


# Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack(
        [
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
        ],
        dim=-1,
    )


# def compl_mul1d(self, a, b):
#     # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
#     op = partial(torch.einsum, "bix,iox->box")
#     return torch.stack(
#         [
#             op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
#             op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
#         ],
#         dim=-1,
#     )

if __name__ == "__main__":
    # Fix the seed
    # torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    batchsize = 3
    in_channels = 8
    out_channels = 8
    modes1 = 11
    length = 20
    x = torch.rand(batchsize, in_channels, length)

    # Do things the Markov way
    weights1 = torch.view_as_real(
        (1 / (in_channels * out_channels))
        * torch.rand(in_channels, out_channels, modes1, dtype=torch.cdouble)
    )
    x_ft = torch.view_as_real(torch.fft.rfft(x))
    print(f"x_ft.shape: {x_ft.shape}")
    out_ft = torch.view_as_real(
        torch.zeros(
            batchsize,
            out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cdouble,
        )
    )
    print(f"out_ft.shape: {out_ft.shape}")
    out_ft[:, :, :modes1] = compl_mul1d(x_ft[:, :, :modes1], weights1)
    # Doing the same with torch.einsum directly
    # out_ft = torch.view_as_complex(out_ft)
    # out_ft[:, :, :modes1] = torch.einsum(
    #     "bix,iox->box",
    #     torch.view_as_complex(x_ft[:, :, :modes1]),
    #     torch.view_as_complex(weights1),
    # )
    # out_ft = torch.view_as_real(out_ft)

    # Do things the Fast Fourier way

    weights_f = copy.deepcopy(weights1)
    out_ft_f = torch.zeros_like(torch.view_as_complex(x_ft))
    print(f"out_ft_f.shape: {out_ft_f.shape}")
    out_ft_f[:, :, :modes1] = torch.einsum(
        "bix,iox->box",
        torch.view_as_complex(x_ft[:, :, :modes1]),
        torch.view_as_complex(weights_f),
    )

    # Compare the two
    print(f"Diff: {torch.max(torch.abs((out_ft - torch.view_as_real(out_ft_f))))}")
    # # a = torch.linspace(0.1, 1.1, 11)
    # # b = torch.linspace(1, 11, 11)
    # # print(f"Difference: {(b-a)/b}")
    # # print(f"Max difference: {torch.max((b-a)/b)}")

    # a = 1e3 * torch.rand(3, 8, 11, dtype=torch.cdouble)
    # b = 1e3 * torch.rand(8, 8, 11, dtype=torch.cdouble)
    # # a = torch.randint(0, 10, (3, 8, 11, 2), dtype=torch.float)
    # # b = torch.randint(0, 10, (8, 8, 11, 2), dtype=torch.float)
    # # a = torch.view_as_complex(a)
    # # b = torch.view_as_complex(b)
    # # a = torch.linspace(1, 24, 24, dtype=torch.cdouble)
    # # print(f"a: {a}")
    # # a = a.reshape(2, 3, 4)
    # # b = torch.linspace(25, 60, 36, dtype=torch.cdouble)
    # # b = b.reshape(3, 3, 4)

    # out1 = torch.einsum("bix,iox->box", a, b)
    # print(f"out1.shape: {out1.shape}")
    # out2 = compl_mul1d(torch.view_as_real(a), torch.view_as_real(b))
    # out2 = torch.view_as_complex(out2)
    # print(f"out2.shape: {out2.shape}")
    # print(f"Diff: {torch.max(torch.abs((out1 - out2)))}")
