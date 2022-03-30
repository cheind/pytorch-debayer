from typing import Tuple

import torch


def peak_signal_noise_ratio(
    x: torch.tensor, y: torch.tensor, datarange: float
) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
    """Computes the channel-wise peak signal to noise ratio (PSNR) for the given image(s).

    Computed according to
        psnr(x,y)   = 10 * log10 (PEAK^2/MSE)
                    = 20 * log10 (PEAK/MSE^0.5)
                    = 20 * log10 PEAK - 10 * log10 MSE
    where PEAK is datarange.

    Args:
        x: (B,C,*) original image
        y: (B,C,*) reconstructed image
        datarange: range of values

    Returns
        psnr: (B,C) peak signal to noise ratios.
        mask: (B,C) boolean mask indicating no differnce.
    """

    B, C = x.shape[:2]
    assert y.shape[:2] == x.shape[:2]

    x = x.reshape(B, C, -1).float()
    y = y.reshape(B, C, -1).float()

    mse = ((x - y) ** 2).mean(-1)  # (B,C)
    # Sanity to return SNR values where there is no error
    mask = mse == 0.0
    mse[mask] = torch.finfo(x.dtype).tiny
    value = 20 * torch.log10(torch.as_tensor(datarange)) - 10 * torch.log10(mse)
    return value, mask
