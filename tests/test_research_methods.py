"""
Automated Unit Tests: Research Methods Module
=============================================
Validates:
- Otsu threshold calculation
- Otsu-Canny Variant (Paper-Inspired)
- Gradient-domain Otsu-Canny
- Threshold ratio constraints and consistency
- Shape and type guarantees
"""

import numpy as np
import pytest

from research_methods import (
    compute_otsu_threshold,
    gradient_otsu_canny,
    otsu_canny_paper_inspired
)


@pytest.fixture
def bimodal_image():
    """Generates an image with distinct bimodal foreground and background distributions."""
    rng = np.random.default_rng(42)
    # Background around 40, foreground around 200
    bg = rng.normal(40, 10, (80, 80))
    fg = rng.normal(200, 10, (80, 80))
    img = np.zeros((80, 80), dtype=np.uint8)
    img[:40, :] = np.clip(bg[:40, :], 0, 255).astype(np.uint8)
    img[40:, :] = np.clip(fg[40:, :], 0, 255).astype(np.uint8)
    return img


def test_compute_otsu_threshold(bimodal_image):
    otsu_val, binarized = compute_otsu_threshold(bimodal_image)
    assert 40 < otsu_val < 200
    assert binarized.shape == bimodal_image.shape
    assert binarized.dtype == np.uint8


def test_otsu_canny_paper_inspired(bimodal_image):
    result = otsu_canny_paper_inspired(bimodal_image, ratio=0.5)

    assert "edges" in result
    assert "otsu_threshold" in result
    assert "low_threshold" in result
    assert "high_threshold" in result
    assert result["variant_name"] == "Otsu-Canny Variant (Paper-Inspired)"

    # Verify that T_low < T_high
    assert result["low_threshold"] < result["high_threshold"]
    # Ratio property
    assert pytest.approx(result["low_threshold"], rel=0.1) == result["high_threshold"] * 0.5
    assert result["edges"].shape == bimodal_image.shape
    assert result["edges"].dtype == np.uint8


def test_gradient_otsu_canny(bimodal_image):
    result = gradient_otsu_canny(bimodal_image, ratio=0.4)

    assert result["edges"].shape == bimodal_image.shape
    assert result["edges"].dtype == np.uint8
    assert result["low_threshold"] < result["high_threshold"]
    assert "Gradient-Domain" in result["variant_name"]
