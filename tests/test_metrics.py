import torch
import math
from debayer.metrics import peak_signal_noise_ratio


def test_psnr():
    o = torch.randint(0, 255, (1, 3, 200, 300)).float()
    r = o.clone()

    # Assert all equal
    psnr, equal_mask = peak_signal_noise_ratio(o, r, 255.0)
    assert equal_mask.all()  #

    # Assert non equal and correct shapes
    r = r + torch.rand_like(o) * 1e-3
    psnr, equal_mask = peak_signal_noise_ratio(o, r, 255.0)
    assert (~equal_mask).all()  #
    assert equal_mask.shape == (1, 3)
    assert psnr.shape == (1, 3)

    # Assert the mean PSNR over channels is the same as for a single channel input
    assert torch.isclose(
        psnr.mean().view(1, 1),
        peak_signal_noise_ratio(o.view(1, 1, -1), r.view(1, 1, -1), 255)[0],
        atol=1e-4,
    )

    # Assert values are correct
    psnr, _ = peak_signal_noise_ratio(
        torch.tensor([10.0]).view(1, 1, 1), torch.tensor([8.0]).view(1, 1, 1), 10.0
    )
    assert torch.isclose(psnr, torch.tensor(20 - 10 * math.log10(4)))
