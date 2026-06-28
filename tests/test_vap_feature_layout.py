import numpy as np

from src.vap_feature_layout import (
    VAP_BC_FEAT_DIM,
    VAP_FEAT_DIM,
    append_bc_tail_features,
    append_bc_tail_to_last_feature,
    bc_tail_summary,
    flat_bc_result,
    flat_vap_result,
)


def test_flat_vap_result_keeps_stable_18_dim_layout():
    result = {
        "p_now": [0.1, 0.2],
        "p_future": [0.3, 0.4],
        "vad": [0.5, 0.6],
        "p_bins": [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
        "p_bins_now": [0.8, 0.9],
        "p_bins_future": [0.11, 0.12],
    }

    feat = flat_vap_result(result)

    assert len(feat) == VAP_FEAT_DIM
    np.testing.assert_allclose(feat[:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(feat[-2:], [0.11, 0.12])


def test_bc_tail_summary_uses_last_max_and_mean_over_tail():
    summary = bc_tail_summary([0.1, 0.2, 0.8, 0.4], frame_rate=2, tail_sec=1.0)

    np.testing.assert_allclose(summary, np.asarray([0.4, 0.8, 0.6], dtype=np.float32))


def test_append_bc_tail_features_is_causal_per_frame():
    vap_feats = np.ones((4, VAP_FEAT_DIM), dtype=np.float32)
    combined = append_bc_tail_features(
        vap_feats,
        [0.1, 0.2, 0.8, 0.4],
        frame_rate=2,
        tail_sec=1.0,
    )

    assert combined.shape == (4, VAP_BC_FEAT_DIM)
    np.testing.assert_allclose(combined[2, -3:], np.asarray([0.8, 0.8, 0.5], dtype=np.float32))
    np.testing.assert_allclose(combined[3, -3:], np.asarray([0.4, 0.8, 0.6], dtype=np.float32))


def test_append_bc_tail_features_zero_fills_missing_bc_values():
    vap_feats = np.ones((2, VAP_FEAT_DIM), dtype=np.float32)

    combined = append_bc_tail_features(vap_feats, [], frame_rate=10, tail_sec=2.0)

    assert combined.shape == (2, VAP_BC_FEAT_DIM)
    np.testing.assert_allclose(combined[:, -3:], np.zeros((2, 3), dtype=np.float32))


def test_append_bc_tail_to_last_feature_matches_test_cache_shape():
    vap_feat = np.ones((VAP_FEAT_DIM,), dtype=np.float32)

    combined = append_bc_tail_to_last_feature(vap_feat, [0.1, 0.7], frame_rate=10)

    assert combined.shape == (VAP_BC_FEAT_DIM,)
    np.testing.assert_allclose(combined[-3:], np.asarray([0.7, 0.7, 0.4], dtype=np.float32))


def test_flat_bc_result_accepts_scalar_like_values():
    assert flat_bc_result({"p_bc": np.asarray([[0.3]], dtype=np.float32)}) == np.float32(0.3)
