"""
Automated Unit Tests: Evaluation Module
=======================================
Validates:
- Execution time measurement precision
- Edge pixel percentage computation and boundary values
- Ground-truth classification metrics (precision, recall, f1)
- Gaussian and salt-and-pepper noise generation
"""

import time
import numpy as np
import pytest

from evaluation import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    calculate_edge_pixel_percentage,
    calculate_ground_truth_metrics,
    measure_execution_time
)


def dummy_processing_function(image: np.ndarray, delay: float = 0.01) -> np.ndarray:
    time.sleep(delay)
    return image.copy()


def test_measure_execution_time():
    img = np.zeros((20, 20), dtype=np.uint8)
    res, elapsed_ms = measure_execution_time(dummy_processing_function, img, delay=0.01)

    assert res.shape == img.shape
    # Elapsed time must be greater than 8ms
    assert elapsed_ms >= 8.0


def test_calculate_edge_pixel_percentage():
    # 10x10 matrix with exactly 25 edge pixels
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:5, :5] = 255

    pct = calculate_edge_pixel_percentage(img, threshold=0)
    assert pct == 25.0

    # Completely blank
    blank = np.zeros((10, 10), dtype=np.uint8)
    assert calculate_edge_pixel_percentage(blank) == 0.0

    # Completely white
    white = np.full((10, 10), 255, dtype=np.uint8)
    assert calculate_edge_pixel_percentage(white) == 100.0


def test_calculate_ground_truth_metrics_exact_match():
    # Exact prediction match with ground truth
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[20:30, 20:30] = 255
    pred = gt.copy()

    metrics = calculate_ground_truth_metrics(pred, gt)
    assert pytest.approx(metrics["precision"], rel=1e-3) == 1.0
    assert pytest.approx(metrics["recall"], rel=1e-3) == 1.0
    assert pytest.approx(metrics["f1_score"], rel=1e-3) == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_calculate_ground_truth_dimension_mismatch():
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((40, 40), dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_ground_truth_metrics(pred, gt)


def test_noise_generation():
    img = np.full((40, 40), 128, dtype=np.uint8)

    # Gaussian noise
    noisy_gauss = add_gaussian_noise(img, sigma=20.0, seed=42)
    assert noisy_gauss.shape == img.shape
    assert noisy_gauss.dtype == np.uint8
    assert not np.array_equal(noisy_gauss, img)

    # Salt & pepper noise
    noisy_sp = add_salt_and_pepper_noise(img, amount=0.1, seed=42)
    assert noisy_sp.shape == img.shape
    assert noisy_sp.dtype == np.uint8
    assert np.count_nonzero(noisy_sp == 255) > 0 or np.count_nonzero(noisy_sp == 0) > 0
