import torch
from src.vap_pool import VapWindowEncoder


def test_output_shape_is_batch_by_hidden():
    enc = VapWindowEncoder(feat_dim=18, hidden=320, conv_channels=64)
    out = enc(torch.randn(4, 20, 18))
    assert out.shape == (4, 320)


def test_length_agnostic():
    enc = VapWindowEncoder(feat_dim=18, hidden=320, conv_channels=64)
    for N in (1, 5, 20, 30):
        assert enc(torch.randn(2, N, 18)).shape == (2, 320)


def test_zeros_input_is_finite():
    enc = VapWindowEncoder(feat_dim=18, hidden=320, conv_channels=64)
    out = enc(torch.zeros(3, 20, 18))
    assert out.shape == (3, 320) and torch.isfinite(out).all()


def test_deterministic_in_eval():
    enc = VapWindowEncoder(feat_dim=18, hidden=16, conv_channels=8).eval()
    x = torch.randn(2, 10, 18)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x))
