"""
Edge Detection Module
=====================
Core implementations of classical edge detection operators:
- Canny Edge Detection (multi-stage: blur, gradient, NMS, double thresholding, hysteresis)
- Sobel Edge Detection (horizontal Gx, vertical Gy, combined magnitude, normalized output)
- Laplacian Edge Detection (second-order derivative, zero-crossing, normalized output)
- Laplacian of Gaussian (LoG) (Gaussian pre-smoothing + Laplacian operator, zero-crossing)
- Prewitt and Roberts Cross Operators (comparison methods)

Authors: College Computer Vision & Image Processing Project
Reference: Sun & Shakhor (2024), DOI: 10.3233/ATDE240501
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Validates input image and converts BGR/RGB to 8-bit single-channel grayscale.

    Args:
        image: Input image array (2D grayscale or 3D color).

    Returns:
        np.ndarray: 2D uint8 grayscale image.

    Raises:
        ValueError: If input is None, empty, or has unsupported dimensions.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid non-empty NumPy array.")
    if image.size == 0:
        raise ValueError("Input image cannot be empty.")

    # Convert floating point images in [0, 1] to [0, 255] uint8 if necessary
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return image.copy()
    elif image.ndim == 3:
        if image.shape[2] == 4:
            # RGBA to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            # BGR to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:
            return image[:, :, 0].copy()
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")


def canny_edge_detection(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    kernel_size: int = 5,
    sigma: float = 1.4,
    apply_blur: bool = True,
    l2_gradient: bool = False
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Performs Canny Edge Detection with parameter validation and detailed stages.

    Canny Algorithm Stages:
    1. Grayscale Conversion: Converts image to single-channel intensity.
    2. Noise Reduction: 2D Gaussian filter removes high-frequency speckle noise.
    3. Gradient Calculation: Computes intensity gradients using Sobel kernels.
    4. Non-Maximum Suppression (NMS): Thins edges by preserving only local gradient maxima.
    5. Double Thresholding: Separates strong edges (>= high_threshold), weak edges
       (between low and high), and non-edges (< low_threshold).
    6. Edge Tracking by Hysteresis: Preserves weak edges only if connected to strong edges.

    Args:
        image: Input image (BGR, RGB, or grayscale).
        low_threshold: Lower hysteresis threshold (default: 50.0).
        high_threshold: Upper hysteresis threshold (default: 150.0).
        kernel_size: Gaussian kernel size (must be positive odd integer, default: 5).
        sigma: Standard deviation for Gaussian blur (default: 1.4).
        apply_blur: Whether to apply Gaussian smoothing prior to Canny (default: True).
        l2_gradient: If True, uses Euclidean L2 norm sqrt(Gx^2 + Gy^2); else L1 |Gx| + |Gy|.

    Returns:
        dict containing:
            - 'edges': Binary edge map (uint8, 0 or 255).
            - 'low_threshold': Validated lower threshold used.
            - 'high_threshold': Validated upper threshold used.
            - 'kernel_size': Gaussian kernel size used.
            - 'smoothed': Gaussian smoothed image (or original grayscale).
    """
    gray = ensure_grayscale(image)

    # Validate and sanitize kernel size
    if kernel_size <= 0:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Validate thresholds
    low_threshold = float(np.clip(low_threshold, 0.0, 255.0))
    high_threshold = float(np.clip(high_threshold, 0.0, 255.0))
    if low_threshold > high_threshold:
        # Swap gracefully if inverted
        low_threshold, high_threshold = high_threshold, low_threshold

    if apply_blur:
        smoothed = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    else:
        smoothed = gray.copy()

    edges = cv2.Canny(
        smoothed,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=3,
        L2gradient=l2_gradient
    )

    return {
        "edges": edges,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "kernel_size": kernel_size,
        "smoothed": smoothed
    }


def sobel_edge_detection(
    image: np.ndarray,
    kernel_size: int = 3,
    threshold: Optional[float] = None,
    normalize: bool = True
) -> Dict[str, Union[np.ndarray, int, Optional[float]]]:
    """
    Performs Sobel Edge Detection computing horizontal (Gx) and vertical (Gy) gradients.

    Theoretical Foundation:
    The Sobel operator is a discrete first-order differential operator that computes
    an approximation of the image intensity gradient.
    Gx computes horizontal changes (vertical edges), Gy computes vertical changes (horizontal edges).
    Combined Gradient Magnitude: G = sqrt(Gx^2 + Gy^2) or approximated as |Gx| + |Gy|.

    Overflow Prevention & Normalization:
    - Intermediate calculations use 64-bit float (`cv2.CV_64F`) to avoid 8-bit integer wrap-around.
    - Gradients are normalized to [0, 255] and converted to uint8 for reliable visual rendering.

    Args:
        image: Input image (BGR, RGB, or grayscale).
        kernel_size: Aperture size for Sobel kernel (must be 1, 3, 5, or 7; default: 3).
        threshold: Optional binarization threshold in [0, 255]. If None, returns continuous gradient.
        normalize: Whether to normalize gradient magnitude to [0, 255] (default: True).

    Returns:
        dict containing:
            - 'edges': Displayable edge image (uint8, continuous or binary).
            - 'grad_x': Normalized absolute horizontal gradient (uint8).
            - 'grad_y': Normalized absolute vertical gradient (uint8).
            - 'magnitude_raw': Raw float64 gradient magnitude.
            - 'kernel_size': Validated kernel size used.
    """
    gray = ensure_grayscale(image)

    # Aperture size must be 1, 3, 5, or 7
    if kernel_size not in (1, 3, 5, 7):
        if kernel_size <= 1:
            kernel_size = 1
        elif kernel_size <= 3:
            kernel_size = 3
        elif kernel_size <= 5:
            kernel_size = 5
        else:
            kernel_size = 7

    # Compute first-order image derivatives in 64-bit float
    grad_x_64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y_64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Compute Euclidean magnitude
    magnitude_raw = np.sqrt(grad_x_64f**2 + grad_y_64f**2)

    # Normalization to [0, 255]
    max_val = magnitude_raw.max()
    if normalize and max_val > 0:
        norm_mag = np.uint8((magnitude_raw / max_val) * 255)
    else:
        norm_mag = np.clip(magnitude_raw, 0, 255).astype(np.uint8)

    grad_x_vis = cv2.convertScaleAbs(grad_x_64f)
    grad_y_vis = cv2.convertScaleAbs(grad_y_64f)

    if threshold is not None:
        thresh_val = float(np.clip(threshold, 0.0, 255.0))
        _, edge_map = cv2.threshold(norm_mag, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        edge_map = norm_mag

    return {
        "edges": edge_map,
        "grad_x": grad_x_vis,
        "grad_y": grad_y_vis,
        "magnitude_raw": magnitude_raw,
        "kernel_size": kernel_size,
        "threshold": threshold
    }


def laplacian_edge_detection(
    image: np.ndarray,
    kernel_size: int = 3,
    pre_blur: bool = False,
    blur_kernel_size: int = 3,
    threshold: Optional[float] = None
) -> Dict[str, Union[np.ndarray, int, Optional[float]]]:
    """
    Performs Laplacian Edge Detection using second-order spatial derivatives.

    Theoretical Foundation:
    The Laplacian operator is defined as:
        nabla^2 f = d^2 f / dx^2 + d^2 f / dy^2
    It highlights regions of rapid intensity change. Unlike gradient operators (first-order),
    the Laplacian responds with a zero-crossing at edge locations.
    Because it calculates second derivatives, it is notoriously sensitive to high-frequency noise.

    Args:
        image: Input image (BGR, RGB, or grayscale).
        kernel_size: Aperture size for Laplacian (must be positive odd integer: 1, 3, 5, 7; default: 3).
        pre_blur: Whether to apply a slight Gaussian pre-filter to mitigate noise sensitivity.
        blur_kernel_size: Size of Gaussian smoothing kernel if pre_blur is True.
        threshold: Optional threshold for binarization of absolute Laplacian response.

    Returns:
        dict containing:
            - 'edges': Displayable edge image (uint8).
            - 'laplacian_raw': Raw float64 Laplacian output (contains positive and negative values).
            - 'kernel_size': Kernel size used.
    """
    gray = ensure_grayscale(image)

    # Validate kernel size
    if kernel_size not in (1, 3, 5, 7):
        kernel_size = 3 if kernel_size < 3 else (5 if kernel_size < 6 else 7)

    if pre_blur:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Calculate second-order derivative using CV_64F
    laplacian_64f = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)

    # Convert to absolute representation and scale to uint8
    abs_laplacian = cv2.convertScaleAbs(laplacian_64f)

    if threshold is not None:
        thresh_val = float(np.clip(threshold, 0.0, 255.0))
        _, edge_map = cv2.threshold(abs_laplacian, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        edge_map = abs_laplacian

    return {
        "edges": edge_map,
        "laplacian_raw": laplacian_64f,
        "kernel_size": kernel_size,
        "pre_blur": pre_blur
    }


def laplacian_of_gaussian(
    image: np.ndarray,
    gaussian_kernel_size: int = 5,
    sigma: float = 1.0,
    laplacian_kernel_size: int = 3,
    zero_crossing: bool = True,
    zero_crossing_threshold: float = 2.0
) -> Dict[str, Union[np.ndarray, int, float]]:
    """
    Performs Laplacian of Gaussian (LoG) Edge Detection.

    Why LoG is distinct from plain Laplacian:
    1. Plain Laplacian is a pure second-derivative operator with zero inherent noise suppression.
    2. LoG first convolves the image with a symmetric Gaussian kernel to smooth out noise
       and scale features, then applies the Laplacian operator.
    3. Edges are accurately identified by finding 'zero-crossings' (points where the
       second derivative transitions from positive to negative or vice versa).

    Algorithm Steps:
    1. Grayscale conversion.
    2. Gaussian smoothing with specified kernel size and sigma.
    3. Laplacian convolution in float64.
    4. Optional zero-crossing detection across 4-connected spatial neighbors with threshold.

    Args:
        image: Input image.
        gaussian_kernel_size: Odd integer kernel size for Gaussian smoothing (default: 5).
        sigma: Standard deviation of Gaussian filter (default: 1.0).
        laplacian_kernel_size: Kernel size for Laplacian operator (1, 3, 5; default: 3).
        zero_crossing: Whether to compute zero-crossing binary edges or normalized response.
        zero_crossing_threshold: Minimum difference across zero to prevent noise triggering (default: 2.0).

    Returns:
        dict containing:
            - 'edges': Resulting edge image (uint8).
            - 'smoothed': Gaussian smoothed intermediate image.
            - 'log_raw': Raw float64 LoG response.
    """
    gray = ensure_grayscale(image)

    # Validate kernel sizes
    if gaussian_kernel_size <= 0:
        gaussian_kernel_size = 5
    if gaussian_kernel_size % 2 == 0:
        gaussian_kernel_size += 1

    if laplacian_kernel_size not in (1, 3, 5, 7):
        laplacian_kernel_size = 3

    # Step 1: Gaussian smoothing
    smoothed = cv2.GaussianBlur(
        gray,
        (gaussian_kernel_size, gaussian_kernel_size),
        sigmaX=sigma,
        sigmaY=sigma
    )

    # Step 2: Laplacian operator on smoothed image
    log_raw = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=laplacian_kernel_size)

    if zero_crossing:
        # Step 3: Zero-crossing detection
        h, w = log_raw.shape
        edge_map = np.zeros((h, w), dtype=np.uint8)

        # Vectorized zero-crossing check between adjacent pixels
        # Horizontal neighbors
        diff_h = log_raw[:, :-1] * log_raw[:, 1:]
        mag_diff_h = np.abs(log_raw[:, :-1] - log_raw[:, 1:])
        cross_h = (diff_h < 0) & (mag_diff_h > zero_crossing_threshold)

        # Vertical neighbors
        diff_v = log_raw[:-1, :] * log_raw[1:, :]
        mag_diff_v = np.abs(log_raw[:-1, :] - log_raw[1:, :])
        cross_v = (diff_v < 0) & (mag_diff_v > zero_crossing_threshold)

        edge_map[:, :-1][cross_h] = 255
        edge_map[:-1, :][cross_v] = 255
    else:
        # Normalized absolute response
        edge_map = cv2.convertScaleAbs(log_raw)

    return {
        "edges": edge_map,
        "smoothed": smoothed,
        "log_raw": log_raw,
        "sigma": sigma,
        "gaussian_kernel_size": gaussian_kernel_size,
        "laplacian_kernel_size": laplacian_kernel_size
    }


def prewitt_edge_detection(
    image: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Union[np.ndarray, Optional[float]]]:
    """
    Performs Prewitt Edge Detection using 3x3 discrete difference kernels.

    Horizontal Kernel (Kx):
        [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]

    Vertical Kernel (Ky):
        [[-1, -1, -1],
         [ 0,  0,  0],
         [ 1,  1,  1]]

    Args:
        image: Input image.
        threshold: Optional binarization threshold.

    Returns:
        dict containing 'edges', 'grad_x', 'grad_y', 'magnitude_raw'.
    """
    gray = ensure_grayscale(image)

    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

    grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_val = magnitude.max()
    norm_mag = np.uint8((magnitude / max_val * 255)) if max_val > 0 else np.zeros_like(gray)

    if threshold is not None:
        thresh_val = float(np.clip(threshold, 0.0, 255.0))
        _, edge_map = cv2.threshold(norm_mag, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        edge_map = norm_mag

    return {
        "edges": edge_map,
        "grad_x": cv2.convertScaleAbs(grad_x),
        "grad_y": cv2.convertScaleAbs(grad_y),
        "magnitude_raw": magnitude
    }


def roberts_cross_edge_detection(
    image: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Union[np.ndarray, Optional[float]]]:
    """
    Performs Roberts Cross Edge Detection using 2x2 diagonal gradient kernels.

    Diagonal Kernel 1 (Gx):
        [[ 1,  0],
         [ 0, -1]]

    Diagonal Kernel 2 (Gy):
        [[ 0,  1],
         [-1,  0]]

    Args:
        image: Input image.
        threshold: Optional binarization threshold.

    Returns:
        dict containing 'edges', 'grad_x', 'grad_y', 'magnitude_raw'.
    """
    gray = ensure_grayscale(image)

    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_val = magnitude.max()
    norm_mag = np.uint8((magnitude / max_val * 255)) if max_val > 0 else np.zeros_like(gray)

    if threshold is not None:
        thresh_val = float(np.clip(threshold, 0.0, 255.0))
        _, edge_map = cv2.threshold(norm_mag, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        edge_map = norm_mag

    return {
        "edges": edge_map,
        "grad_x": cv2.convertScaleAbs(grad_x),
        "grad_y": cv2.convertScaleAbs(grad_y),
        "magnitude_raw": magnitude
    }
