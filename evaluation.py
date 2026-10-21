"""
Evaluation Module
=================
Performance measurement, edge density calculation, noise injection, and ground-truth validation.

Metrics Principles:
-------------------
1. Execution Time: Wall-clock latency in milliseconds measured via `time.perf_counter()`.
   Indicates computational efficiency, NOT edge quality or accuracy.
2. Edge Pixel Percentage: (count of pixels > threshold / total pixels) * 100.
   Indicates edge sparsity or density. A higher edge percentage does NOT imply superior detection;
   noisy detectors (like unblurred Laplacian) often mark extensive noise as edges, yielding high
   percentages with poor semantic accuracy.
3. Ground-Truth Validation: Precision, Recall, and F1-Score are only computed when a genuine,
   spatially aligned ground-truth binary mask is supplied. No synthetic or fabricated accuracy
   metrics are generated in the absence of verified ground truth.
"""

import time
from typing import Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def measure_execution_time(
    func: Callable,
    *args,
    **kwargs
) -> Tuple[any, float]:
    """
    Executes a function and measures its wall-clock runtime in milliseconds.

    Args:
        func: The image processing callable.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        Tuple[any, float]: (function_result, elapsed_milliseconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return result, elapsed_ms


def calculate_edge_pixel_percentage(
    edge_map: np.ndarray,
    threshold: int = 0
) -> float:
    """
    Calculates the percentage of pixels classified as edges.

    Definition:
    -----------
    A pixel is considered an edge if its intensity is strictly greater than `threshold`.
    Edge Pixel Percentage (%) = (Number of Edge Pixels / Total Pixels) * 100.0

    Note:
    -----
    This metric reflects detection density. It is NOT an accuracy score.
    A noisy image may produce a high edge percentage despite being incorrect.

    Args:
        edge_map: 2D or 3D image array.
        threshold: Intensity cut-off (default: 0).

    Returns:
        float: Percentage of edge pixels in [0.0, 100.0].
    """
    if edge_map is None or edge_map.size == 0:
        return 0.0

    if edge_map.ndim == 3:
        edge_map = edge_map[:, :, 0]

    total_pixels = edge_map.size
    if total_pixels == 0:
        return 0.0

    edge_pixels = np.count_nonzero(edge_map > threshold)
    return float((edge_pixels / total_pixels) * 100.0)


def calculate_ground_truth_metrics(
    predicted_edge_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: int = 127,
    tolerance_radius: int = 0
) -> Dict[str, float]:
    """
    Calculates objective classification metrics against a verified ground-truth edge mask.

    Definitions:
    - True Positives (TP): Pixels marked as edge in both prediction and ground truth.
    - False Positives (FP): Pixels marked as edge in prediction but background in ground truth.
    - False Negatives (FN): Pixels marked as background in prediction but edge in ground truth.
    - Precision: TP / (TP + FP + eps)
    - Recall: TP / (TP + FN + eps)
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall + eps)

    Args:
        predicted_edge_map: Binary or continuous edge detection output.
        ground_truth_mask: Binary ground truth edge mask.
        threshold: Binarization cut-off for prediction and ground truth.
        tolerance_radius: Morphological dilation radius in pixels for spatial matching (default: 0).

    Returns:
        dict containing 'precision', 'recall', 'f1_score', 'true_positives', 'false_positives', 'false_negatives'.
    """
    if predicted_edge_map is None or ground_truth_mask is None:
        raise ValueError("Both predicted edge map and ground truth mask must be provided.")

    if predicted_edge_map.shape[:2] != ground_truth_mask.shape[:2]:
        raise ValueError(
            f"Dimension mismatch: Prediction {predicted_edge_map.shape[:2]} vs Ground Truth {ground_truth_mask.shape[:2]}"
        )

    pred_bin = (predicted_edge_map > threshold).astype(np.uint8)
    gt_bin = (ground_truth_mask > threshold).astype(np.uint8)

    if tolerance_radius > 0:
        # Spatial tolerance via morphological dilation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2 * tolerance_radius + 1, 2 * tolerance_radius + 1)
        )
        gt_dilated = cv2.dilate(gt_bin, kernel)
        pred_dilated = cv2.dilate(pred_bin, kernel)

        tp = np.count_nonzero(pred_bin & gt_dilated)
        fp = np.count_nonzero(pred_bin & ~gt_dilated)
        fn = np.count_nonzero(gt_bin & ~pred_dilated)
    else:
        tp = np.count_nonzero(pred_bin & gt_bin)
        fp = np.count_nonzero(pred_bin & ~gt_bin)
        fn = np.count_nonzero(~pred_bin & gt_bin)

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Adds zero-mean Gaussian noise to an image with controlled reproducible seed.

    Args:
        image: Input image.
        mean: Gaussian mean (default: 0.0).
        sigma: Standard deviation of Gaussian distribution (default: 25.0).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        np.ndarray: Noisy image clipped to [0, 255] uint8.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    gauss = rng.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Adds impulse (salt and pepper) noise to an image.

    Args:
        image: Input image.
        amount: Proportion of total image pixels corrupted by noise in [0.0, 1.0].
        salt_vs_pepper: Ratio of salt (white 255) vs pepper (black 0) in [0.0, 1.0].
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Noisy image uint8.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    noisy = image.copy()
    num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * image.size * (1.0 - salt_vs_pepper)))

    # Coordinates for salt
    coords_salt = [rng.integers(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords_salt)] = 255

    # Coordinates for pepper
    coords_pepper = [rng.integers(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords_pepper)] = 0

    return noisy
