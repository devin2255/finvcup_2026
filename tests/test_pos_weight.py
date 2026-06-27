import numpy as np
from src.pos_weight import compute_pos_weight


def test_global_cap_applied():
    # label0: pos=1,neg=99 -> raw 99 capped to 8; label1: pos=50,neg=50 -> 1.0
    y = np.zeros((100, 2), np.float32); y[:1, 0] = 1; y[:50, 1] = 1
    pw = compute_pos_weight(y, ["bc", "t"], cap=8.0, per_label_cap=None)
    np.testing.assert_allclose(pw, [8.0, 1.0], rtol=1e-5)


def test_per_label_cap_overrides_for_bc():
    y = np.zeros((100, 2), np.float32); y[:1, 0] = 1; y[:1, 1] = 1   # both raw 99
    pw = compute_pos_weight(y, ["bc", "t"], cap=8.0, per_label_cap={"bc": 16.0})
    np.testing.assert_allclose(pw, [16.0, 8.0], rtol=1e-5)   # bc->16, t->8


def test_zero_pos_no_div_by_zero():
    y = np.zeros((10, 2), np.float32); y[:5, 1] = 1          # label0 has no positives
    pw = compute_pos_weight(y, ["bc", "t"], cap=8.0, per_label_cap=None)
    assert np.isfinite(pw).all() and pw[0] == 8.0            # capped, finite


def test_per_label_cap_none_equals_global():
    y = np.zeros((100, 3), np.float32); y[:2, :] = 1
    a = compute_pos_weight(y, ["a", "b", "c"], cap=8.0, per_label_cap=None)
    b = compute_pos_weight(y, ["a", "b", "c"], cap=8.0, per_label_cap={})
    np.testing.assert_array_equal(a, b)
