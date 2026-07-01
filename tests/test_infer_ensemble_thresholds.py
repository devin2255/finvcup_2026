import numpy as np

from src.inference_utils import combine_threshold_vectors


def test_combine_threshold_vectors_supports_mean_best_and_weighted():
    thresholds = [
        np.array([0.2, 0.4], dtype=np.float32),
        np.array([0.4, 0.8], dtype=np.float32),
    ]
    metrics = [1.0, 3.0]

    np.testing.assert_allclose(
        combine_threshold_vectors(thresholds, metrics, "mean"),
        np.array([0.3, 0.6], dtype=np.float64),
    )
    np.testing.assert_allclose(
        combine_threshold_vectors(thresholds, metrics, "best"),
        np.array([0.2, 0.4], dtype=np.float64),
    )
    np.testing.assert_allclose(
        combine_threshold_vectors(thresholds, metrics, "weighted_mean"),
        np.array([0.35, 0.7], dtype=np.float64),
    )
