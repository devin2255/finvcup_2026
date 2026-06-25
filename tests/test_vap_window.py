import numpy as np
from src.vap_window import _extract_vap_window


def test_normal_window_is_last_n_inclusive_of_fr():
    arr = np.arange(100 * 18, dtype=np.float32).reshape(100, 18)
    w = _extract_vap_window(arr, fr=50, N=20)
    assert w.shape == (20, 18)
    np.testing.assert_array_equal(w, arr[31:51])  # inclusive of fr=50


def test_near_start_left_replicate_pads_earliest_frame():
    arr = np.arange(10 * 18, dtype=np.float32).reshape(10, 18)
    w = _extract_vap_window(arr, fr=5, N=20)
    assert w.shape == (20, 18)
    np.testing.assert_array_equal(w[-6:], arr[0:6])     # real frames at the end
    for i in range(20 - 6):                              # left pad = replicate arr[0]
        np.testing.assert_array_equal(w[i], arr[0])


def test_none_returns_zeros():
    w = _extract_vap_window(None, fr=0, N=20)
    assert w.shape == (20, 18) and not w.any()


def test_empty_returns_zeros():
    w = _extract_vap_window(np.zeros((0, 18), np.float32), fr=0, N=20)
    assert w.shape == (20, 18) and not w.any()


def test_fr_clamped_above_range():
    arr = np.arange(5 * 18, dtype=np.float32).reshape(5, 18)
    w = _extract_vap_window(arr, fr=999, N=3)
    np.testing.assert_array_equal(w, arr[2:5])           # fr clamps to 4


def test_old_1d_single_frame_treated_as_one_frame():
    arr = np.arange(18, dtype=np.float32)                 # shape (18,)
    w = _extract_vap_window(arr, fr=0, N=4)
    assert w.shape == (4, 18)
    for i in range(4):
        np.testing.assert_array_equal(w[i], arr)


def test_train_and_test_paths_agree_on_same_frames():
    arr = np.random.RandomState(0).randn(60, 18).astype(np.float32)
    fr, N = 40, 20
    train_w = _extract_vap_window(arr, fr=fr, N=N)
    cache = arr[fr - N + 1: fr + 1]                       # what precompute saves: [N,18]
    test_w = _extract_vap_window(cache, fr=cache.shape[0] - 1, N=N)
    np.testing.assert_array_equal(train_w, test_w)
