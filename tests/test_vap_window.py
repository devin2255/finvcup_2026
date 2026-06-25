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


# --- Task 2: vap_last_n_frames streaming aggregator ---
from src.vap_window import vap_last_n_frames, _flat_result


class _FakeQueue:
    def __init__(self): self._items = []
    def empty(self): return not self._items
    def get(self): return self._items.pop(0)
    def put(self, x): self._items.append(x)


def _frame_result(k: int) -> dict:
    # vad[0] = k acts as a frame-index tag; _flat_result puts vad at index 4
    return {"p_now": [0.0, 0.0], "p_future": [0.0, 0.0], "vad": [float(k), 0.0],
            "p_bins": [[0, 0, 0, 0], [0, 0, 0, 0]],
            "p_bins_now": [0.0, 0.0], "p_bins_future": [0.0, 0.0]}


class _FakeMaai:
    """Emits exactly one result dict per process() call, tagged by call index."""
    def __init__(self): self.result_dict_queue = _FakeQueue(); self._i = 0
    def reset_runtime_state(self): self._i = 0
    def process(self, c1, c2):
        self.result_dict_queue.put(_frame_result(self._i)); self._i += 1


def test_flat_result_is_18_dims_with_vad_tag():
    v = _flat_result(_frame_result(7))
    assert len(v) == 18 and v[4] == 7.0


def test_keeps_last_n_frames_when_more_than_n():
    fs, M, N = 4, 10, 5
    audio2 = np.zeros((2, fs * M), dtype=np.float32)     # 10 process() calls
    out = vap_last_n_frames(_FakeMaai(), audio2, frame_samples=fs, N=N)
    assert out.shape == (N, 18)
    np.testing.assert_array_equal(out[:, 4], np.array([5, 6, 7, 8, 9], np.float32))


def test_left_pads_when_fewer_than_n():
    fs, M, N = 4, 3, 5
    audio2 = np.zeros((2, fs * M), dtype=np.float32)     # 3 process() calls -> frames 0,1,2
    out = vap_last_n_frames(_FakeMaai(), audio2, frame_samples=fs, N=N)
    assert out.shape == (N, 18)
    np.testing.assert_array_equal(out[:, 4], np.array([0, 0, 0, 1, 2], np.float32))


def test_empty_audio_returns_zeros():
    out = vap_last_n_frames(_FakeMaai(), np.zeros((2, 0), np.float32), frame_samples=4, N=5)
    assert out.shape == (5, 18) and not out.any()
