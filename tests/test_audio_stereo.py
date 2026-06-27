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


# --- Task 2: StereoActivityEncoder ---
from src.audio_stereo import StereoActivityEncoder


def _enc(**kw):
    # small + cheap defaults for CPU tests
    return StereoActivityEncoder(sample_rate=16000, n_mels=32,
                                 conv_channels=(16, 32, 48), tail_sec=1.0, dropout=0.0, **kw)


def test_output_shape_is_batch_by_out_dim():
    enc = _enc().eval()
    out = enc(torch.randn(2, 2, 32000))   # 2s stereo
    assert out.shape == (2, 48)
    assert enc.out_dim == 48


def test_uses_only_tail_window():
    # tail_sec=1.0 -> last 16000 samples; changing the pre-tail region must not change output
    enc = _enc().eval()
    x = torch.randn(1, 2, 32000)
    x2 = x.clone()
    x2[..., :16000] = torch.randn(1, 2, 16000)   # mutate only the discarded prefix
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x2), atol=1e-5)


def test_short_input_shorter_than_tail_still_works():
    enc = _enc().eval()
    out = enc(torch.randn(2, 2, 8000))    # 0.5s < tail
    assert out.shape == (2, 48)


def test_zeros_input_is_finite():
    enc = _enc().eval()
    with torch.no_grad():
        out = enc(torch.zeros(2, 2, 32000))
    assert out.shape == (2, 48) and torch.isfinite(out).all()


def test_deterministic_in_eval():
    enc = _enc().eval()
    x = torch.randn(2, 2, 24000)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x))
