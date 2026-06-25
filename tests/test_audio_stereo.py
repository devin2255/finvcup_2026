import torch
from src.audio_stereo import _slice_tail


def test_slice_returns_last_tail_samples_when_longer():
    wave = torch.arange(2 * 2 * 100, dtype=torch.float32).reshape(2, 2, 100)
    out = _slice_tail(wave, 30)
    assert out.shape == (2, 2, 30)
    assert torch.equal(out, wave[..., -30:])


def test_slice_returns_full_when_shorter_or_equal():
    wave = torch.randn(2, 2, 20)
    assert torch.equal(_slice_tail(wave, 50), wave)   # shorter than tail
    assert torch.equal(_slice_tail(wave, 20), wave)   # equal to tail


def test_slice_nonpositive_or_none_returns_full():
    wave = torch.randn(1, 2, 10)
    assert torch.equal(_slice_tail(wave, 0), wave)
    assert torch.equal(_slice_tail(wave, None), wave)
