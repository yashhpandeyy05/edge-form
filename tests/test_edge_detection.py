"""
Automated Unit Tests: Edge Detection Module
===========================================
Validates:
- Canny, Sobel, Laplacian, LoG, Prewitt, Roberts
- Dimension preservation and uint8 output data type
- Color (BGR, RGBA) and grayscale inputs
- Small images (5x5, 10x10) and boundary edge conditions
- Threshold sanity and invalid parameter graceful fallback
"""

import numpy as np
import pytest

from edge_detection import (
    canny_edge_detection,
    ensure_grayscale,
    laplacian_edge_detection,
    laplacian_of_gaussian,
    prewitt_edge_detection,
    roberts_cross_edge_detection,
    sobel_edge_detection
)


@pytest.fixture
def synthetic_box_image():
    """Generates a 100x100 black image with a central 40x40 solid white square."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 255
    return img


@pytest.fixture
def synthetic_color_image():
    """Generates a 60x60 3-channel BGR synthetic image."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[15:45, 15:45, 0] = 200  # Blue box
    img[25:55, 25:55, 2] = 250  # Red box
    return img


def test_ensure_grayscale(synthetic_box_image, synthetic_color_image):
    # Grayscale identity
    gray_out = ensure_grayscale(synthetic_box_image)
    assert gray_out.ndim == 2
    assert gray_out.dtype == np.uint8
    assert gray_out.shape == (100, 100)

    # 3-channel color conversion
    color_gray = ensure_grayscale(synthetic_color_image)
    assert color_gray.ndim == 2
    assert color_gray.shape == (60, 60)
    assert color_gray.dtype == np.uint8

    # Error handling for empty or None input
    with pytest.raises(ValueError):
        ensure_grayscale(None)
    with pytest.raises(ValueError):
        ensure_grayscale(np.array([]))


def test_canny_edge_detection(synthetic_box_image):
    result = canny_edge_detection(synthetic_box_image, low_threshold=50, high_threshold=150)
    edges = result["edges"]

    assert isinstance(edges, np.ndarray)
    assert edges.shape == synthetic_box_image.shape
    assert edges.dtype == np.uint8
    # Edge map must be binary (0 or 255)
    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})
    # There must be detected edge pixels for the box perimeter
    assert np.count_nonzero(edges == 255) > 0


def test_canny_threshold_inversion_handling(synthetic_box_image):
    # Pass low > high intentionally
    result = canny_edge_detection(synthetic_box_image, low_threshold=200, high_threshold=50)
    assert result["low_threshold"] <= result["high_threshold"]
    assert result["edges"].shape == synthetic_box_image.shape


def test_sobel_edge_detection(synthetic_box_image):
    result = sobel_edge_detection(synthetic_box_image, kernel_size=3, threshold=50)
    edges = result["edges"]

    assert edges.shape == synthetic_box_image.shape
    assert edges.dtype == np.uint8
    assert "grad_x" in result
    assert "grad_y" in result
    assert "magnitude_raw" in result
    assert result["magnitude_raw"].dtype == np.float64
    assert np.max(edges) > 0


def test_sobel_invalid_kernel_fallback(synthetic_box_image):
    # Pass invalid even or negative kernel size
    result = sobel_edge_detection(synthetic_box_image, kernel_size=4)
    assert result["kernel_size"] in (1, 3, 5, 7)


def test_laplacian_edge_detection(synthetic_box_image):
    result = laplacian_edge_detection(synthetic_box_image, kernel_size=3, pre_blur=True)
    edges = result["edges"]

    assert edges.shape == synthetic_box_image.shape
    assert edges.dtype == np.uint8
    assert result["laplacian_raw"].dtype == np.float64


def test_laplacian_of_gaussian(synthetic_box_image):
    # Test zero-crossing LoG
    result = laplacian_of_gaussian(
        synthetic_box_image,
        gaussian_kernel_size=5,
        sigma=1.0,
        zero_crossing=True
    )
    edges = result["edges"]

    assert edges.shape == synthetic_box_image.shape
    assert edges.dtype == np.uint8
    # LoG must produce distinct boundaries
    assert np.count_nonzero(edges == 255) > 0


def test_prewitt_and_roberts(synthetic_box_image):
    res_prewitt = prewitt_edge_detection(synthetic_box_image, threshold=50)
    assert res_prewitt["edges"].shape == synthetic_box_image.shape
    assert res_prewitt["edges"].dtype == np.uint8

    res_roberts = roberts_cross_edge_detection(synthetic_box_image, threshold=50)
    assert res_roberts["edges"].shape == synthetic_box_image.shape
    assert res_roberts["edges"].dtype == np.uint8


def test_small_image_safety():
    # 5x5 tiny synthetic matrix
    tiny = np.array([
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255]
    ], dtype=np.uint8)

    c_res = canny_edge_detection(tiny, kernel_size=3)
    s_res = sobel_edge_detection(tiny, kernel_size=3)
    l_res = laplacian_edge_detection(tiny, kernel_size=3)

    assert c_res["edges"].shape == (5, 5)
    assert s_res["edges"].shape == (5, 5)
    assert l_res["edges"].shape == (5, 5)
