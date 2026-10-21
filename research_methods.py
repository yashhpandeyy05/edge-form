"""
Research Methods Module
=======================
Implementation of research-referenced and research-inspired adaptive edge detection.

Reference Paper:
----------------
Title: "Performance Investigating of Several Classical Edge Detection Operator for Segmentation"
Authors: Lei Sun and Shariar Shakhor
Publication: Advances in Transdisciplinary Engineering, Vol. 50, pp. 497-506, 2024
DOI: https://doi.org/10.3233/ATDE240501

Research Context & Integrity Disclosure:
----------------------------------------
In Sun & Shakhor (2024), the authors investigate classical edge detection operators
(Canny, Sobel, Laplacian, Prewitt, Roberts) for segmentation tasks. A core challenge
highlighted in their analysis is the sensitivity of the Canny algorithm to arbitrary,
empirically chosen dual thresholds (T_low, T_high). The paper explores combining
Otsu's thresholding principle to establish automatic edge thresholds.

Because the full proprietary conference proceedings do not disclose an exact closed-form
hyperparameter ratio for all edge maps, we distinguish:
1. "Verified Paper Principles":
   - Use of Otsu's discriminant criterion to compute an objective, data-driven threshold.
   - Elimination of subjective manual trial-and-error in Canny thresholding.
2. "Otsu-Canny Variant (Paper-Inspired Approximation)":
   - Explicitly labelled implementation where T_high is set to the Otsu threshold T_otsu,
     and T_low is computed as (ratio * T_otsu) (default ratio = 0.5).
   - A complementary "Gradient-Domain Otsu-Canny" that computes Otsu directly on the
     Sobel gradient magnitude map, directly matching Canny's gradient threshold space.
"""

from typing import Dict, Tuple, Union
import cv2
import numpy as np

from edge_detection import ensure_grayscale


def compute_otsu_threshold(image: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes Otsu's optimal global threshold using maximum inter-class variance.

    Mathematical Formulation:
    -------------------------
    Otsu's method searches for the threshold t* that maximizes the between-class variance:
        sigma_b^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2
    where:
        - omega_0(t), omega_1(t) are class probabilities for background and foreground.
        - mu_0(t), mu_1(t) are the mean intensities of background and foreground.

    Data Operated On:
    -----------------
    Operates on the 1D intensity histogram of an 8-bit single-channel image.

    Args:
        image: Single-channel uint8 grayscale image.

    Returns:
        Tuple[float, np.ndarray]:
            - otsu_val: The calculated optimal threshold value.
            - binarized: The thresholded binary image.
    """
    gray = ensure_grayscale(image)
    otsu_val, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(otsu_val), binarized


def otsu_canny_paper_inspired(
    image: np.ndarray,
    ratio: float = 0.5,
    blur_kernel_size: int = 5,
    sigma: float = 1.4,
    apply_blur: bool = True
) -> Dict[str, Union[np.ndarray, float, int, str]]:
    """
    Otsu-Canny Variant (Paper-Inspired Approximation)
    ================================================
    Automatically determines Canny hysteresis thresholds using Otsu's method
    applied to the intensity distribution of the image.

    Procedure:
    ----------
    1. Pre-process input image into grayscale.
    2. Apply Gaussian smoothing with specified kernel and sigma if apply_blur is True.
    3. Calculate global Otsu threshold T_otsu from the smoothed grayscale intensity histogram.
    4. Derive upper hysteresis threshold: T_high = T_otsu.
    5. Derive lower hysteresis threshold: T_low = ratio * T_high (default ratio = 0.5).
    6. Execute Canny edge detection using [T_low, T_high].

    Comparison with Manual Canny:
    -----------------------------
    - Manual Canny requires user intervention and varies widely across lighting conditions.
    - Otsu-Canny is adaptive and deterministic, computing thresholds directly from image statistics.

    Assumptions & Limitations:
    --------------------------
    - Bimodal Assumption: Otsu assumes the intensity histogram is bimodal (distinct foreground/background).
      If the image has uniform gradients, complex textures, or heavy illumination gradients,
      the intensity Otsu threshold may not directly correspond to edge prominence.
    - Intensity vs. Gradient Space: Standard Otsu operates on pixel intensities, whereas Canny's
      hysteresis thresholds filter gradient magnitudes. The ratio scaling bridges this gap
      in a standardized manner.

    Args:
        image: Input image (BGR, RGB, or grayscale).
        ratio: Ratio between lower and upper threshold (default: 0.5).
        blur_kernel_size: Size of Gaussian blur kernel (default: 5).
        sigma: Gaussian smoothing standard deviation (default: 1.4).
        apply_blur: Whether to smooth prior to threshold computation and Canny.

    Returns:
        dict containing:
            - 'edges': Resulting binary edge map (uint8).
            - 'otsu_threshold': Raw Otsu threshold computed.
            - 'low_threshold': Computed lower threshold (T_low).
            - 'high_threshold': Computed upper threshold (T_high).
            - 'ratio': Ratio used.
            - 'variant_name': "Otsu-Canny Variant (Paper-Inspired)".
    """
    gray = ensure_grayscale(image)

    if blur_kernel_size <= 0:
        blur_kernel_size = 5
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    if apply_blur:
        smoothed = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), sigmaX=sigma, sigmaY=sigma)
    else:
        smoothed = gray.copy()

    # Step 1: Compute Otsu threshold from intensity distribution
    otsu_val, _ = compute_otsu_threshold(smoothed)

    # Step 2: Establish hysteresis thresholds
    ratio = float(np.clip(ratio, 0.05, 0.95))
    high_threshold = float(np.clip(otsu_val, 10.0, 245.0))
    low_threshold = float(np.clip(ratio * high_threshold, 5.0, 240.0))

    # Step 3: Run Canny
    edges = cv2.Canny(
        smoothed,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=3,
        L2gradient=False
    )

    return {
        "edges": edges,
        "otsu_threshold": otsu_val,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "ratio": ratio,
        "blur_kernel_size": blur_kernel_size,
        "variant_name": "Otsu-Canny Variant (Paper-Inspired)",
        "method_domain": "Intensity Distribution"
    }


def gradient_otsu_canny(
    image: np.ndarray,
    ratio: float = 0.5,
    blur_kernel_size: int = 5,
    sigma: float = 1.4
) -> Dict[str, Union[np.ndarray, float, int, str]]:
    """
    Gradient-Domain Otsu-Canny (Advanced Research Extension)
    =======================================================
    Computes Otsu's threshold directly on the Sobel gradient magnitude histogram
    instead of the grayscale intensity histogram.

    Theoretical Motivation:
    -----------------------
    Because Canny's double thresholds act on the spatial gradient magnitude image |nabla I|,
    applying Otsu directly to the normalized gradient magnitude naturally separates
    the "edge class" (high gradients) from the "flat background class" (low gradients).

    Args:
        image: Input image.
        ratio: Ratio between lower and upper threshold (default: 0.5).
        blur_kernel_size: Gaussian kernel size.
        sigma: Gaussian smoothing standard deviation.

    Returns:
        dict containing 'edges', 'otsu_threshold', 'low_threshold', 'high_threshold', etc.
    """
    gray = ensure_grayscale(image)

    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    smoothed = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), sigmaX=sigma, sigmaY=sigma)

    # Compute first-order gradient magnitude via Sobel
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)

    max_mag = mag.max()
    if max_mag > 0:
        norm_mag = np.uint8((mag / max_mag) * 255)
    else:
        norm_mag = np.zeros_like(gray)

    # Compute Otsu directly on gradient magnitudes
    otsu_val, _ = compute_otsu_threshold(norm_mag)

    ratio = float(np.clip(ratio, 0.05, 0.95))
    high_threshold = float(np.clip(otsu_val, 15.0, 240.0))
    low_threshold = float(np.clip(ratio * high_threshold, 5.0, 230.0))

    edges = cv2.Canny(
        smoothed,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=3,
        L2gradient=False
    )

    return {
        "edges": edges,
        "otsu_threshold": otsu_val,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "ratio": ratio,
        "blur_kernel_size": blur_kernel_size,
        "variant_name": "Gradient-Domain Otsu-Canny (Research Extension)",
        "method_domain": "Gradient Magnitude Distribution"
    }
