"""
Pipeline runner script for API and CLI integration.
Processes images with edge detection operators and outputs JSON results with base64 encoded images.
"""

import base64
import json
import sys
import cv2
import numpy as np

from edge_detection import (
    canny_edge_detection,
    ensure_grayscale,
    laplacian_edge_detection,
    laplacian_of_gaussian,
    prewitt_edge_detection,
    roberts_cross_edge_detection,
    sobel_edge_detection
)
from evaluation import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    calculate_edge_pixel_percentage,
    measure_execution_time
)
from research_methods import (
    compute_otsu_threshold,
    gradient_otsu_canny,
    otsu_canny_paper_inspired
)


def img_to_base64(arr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def handle_request(payload: dict) -> dict:
    action = payload.get("action", "process")

    # Load image: either base64 or file path
    if "image_base64" in payload and payload["image_base64"]:
        b64_str = payload["image_base64"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif "image_path" in payload and payload["image_path"]:
        img = cv2.imread(payload["image_path"], cv2.IMREAD_COLOR)
    else:
        # Default fallback to sample
        img = cv2.imread("sample_images/geometric_shapes.png", cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(img, (40, 40), (160, 160), (255, 255, 255), -1)

    if img is None:
        raise ValueError("Could not decode or load input image.")

    # Apply noise if requested
    noise_type = payload.get("noise_type", "none")
    if noise_type == "gaussian":
        sigma = float(payload.get("noise_sigma", 25.0))
        img = add_gaussian_noise(img, sigma=sigma, seed=42)
    elif noise_type == "salt_and_pepper":
        amt = float(payload.get("noise_amount", 0.05))
        img = add_salt_and_pepper_noise(img, amount=amt, seed=42)

    original_b64 = img_to_base64(img)
    gray = ensure_grayscale(img)
    gray_b64 = img_to_base64(gray)

    if action == "process":
        algo = payload.get("algorithm", "Canny")
        params = payload.get("params", {})

        if algo == "Canny":
            res, elapsed = measure_execution_time(
                canny_edge_detection, img,
                low_threshold=float(params.get("low_threshold", 50.0)),
                high_threshold=float(params.get("high_threshold", 150.0)),
                kernel_size=int(params.get("kernel_size", 5)),
                sigma=float(params.get("sigma", 1.4)),
                apply_blur=bool(params.get("apply_blur", True)),
                l2_gradient=bool(params.get("l2_gradient", False))
            )
            edge_map = res["edges"]

        elif algo == "Sobel":
            thresh = float(params["threshold"]) if params.get("apply_threshold", False) else None
            sobel_mode = params.get("sobel_mode", "combined")
            res, elapsed = measure_execution_time(
                sobel_edge_detection, img,
                kernel_size=int(params.get("kernel_size", 3)),
                threshold=thresh,
                normalize=bool(params.get("normalize", True))
            )
            if sobel_mode == "horizontal":
                edge_map = res["grad_x"]
            elif sobel_mode == "vertical":
                edge_map = res["grad_y"]
            else:
                edge_map = res["edges"]

        elif algo == "Laplacian":
            thresh = float(params["threshold"]) if params.get("apply_threshold", False) else None
            res, elapsed = measure_execution_time(
                laplacian_edge_detection, img,
                kernel_size=int(params.get("kernel_size", 3)),
                pre_blur=bool(params.get("pre_blur", False)),
                blur_kernel_size=int(params.get("blur_kernel_size", 3)),
                threshold=thresh
            )
            edge_map = res["edges"]

        elif algo == "LoG":
            res, elapsed = measure_execution_time(
                laplacian_of_gaussian, img,
                gaussian_kernel_size=int(params.get("gaussian_kernel_size", 5)),
                sigma=float(params.get("sigma", 1.0)),
                laplacian_kernel_size=int(params.get("laplacian_kernel_size", 3)),
                zero_crossing=bool(params.get("zero_crossing", True)),
                zero_crossing_threshold=float(params.get("zero_crossing_threshold", 2.0))
            )
            edge_map = res["edges"]

        elif algo == "Prewitt":
            thresh = float(params["threshold"]) if params.get("apply_threshold", False) else None
            res, elapsed = measure_execution_time(prewitt_edge_detection, img, threshold=thresh)
            edge_map = res["edges"]

        elif algo == "Roberts":
            thresh = float(params["threshold"]) if params.get("apply_threshold", False) else None
            res, elapsed = measure_execution_time(roberts_cross_edge_detection, img, threshold=thresh)
            edge_map = res["edges"]

        elif algo == "Otsu-Canny":
            ratio = float(params.get("ratio", 0.5))
            sigma = float(params.get("sigma", 1.4))
            res, elapsed = measure_execution_time(
                otsu_canny_paper_inspired, img,
                ratio=ratio, sigma=sigma
            )
            edge_map = res["edges"]
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        density = calculate_edge_pixel_percentage(edge_map)
        edge_b64 = img_to_base64(edge_map)

        response_dict = {
            "status": "success",
            "algorithm": algo,
            "original_image": original_b64,
            "gray_image": gray_b64,
            "edge_image": edge_b64,
            "latency_ms": round(elapsed, 2),
            "edge_density_pct": round(density, 2),
            "width": img.shape[1],
            "height": img.shape[0],
            "extra": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                      for k, v in res.items() if not isinstance(v, np.ndarray)}
        }
        if "grad_x" in res:
            response_dict["grad_x_image"] = img_to_base64(res["grad_x"])
        if "grad_y" in res:
            response_dict["grad_y_image"] = img_to_base64(res["grad_y"])

        return response_dict

    elif action == "benchmark":
        # Run comparative benchmark across all operators
        methods = [
            ("Canny (Manual 50/150)", lambda x: canny_edge_detection(x, 50, 150)["edges"]),
            ("Otsu-Canny (Paper-Inspired)", lambda x: otsu_canny_paper_inspired(x, ratio=0.5)["edges"]),
            ("Gradient-Domain Otsu-Canny", lambda x: gradient_otsu_canny(x, ratio=0.5)["edges"]),
            ("Sobel Magnitude", lambda x: sobel_edge_detection(x, kernel_size=3)["edges"]),
            ("Laplacian", lambda x: laplacian_edge_detection(x, kernel_size=3, pre_blur=False)["edges"]),
            ("Laplacian of Gaussian (LoG)", lambda x: laplacian_of_gaussian(x, 5, 1.0, 3, zero_crossing=True)["edges"]),
            ("Prewitt", lambda x: prewitt_edge_detection(x)["edges"]),
            ("Roberts Cross", lambda x: roberts_cross_edge_detection(x)["edges"])
        ]

        benchmarks = []
        images_dict = {}

        for name, fn in methods:
            edge_res, elapsed = measure_execution_time(fn, img)
            pct = calculate_edge_pixel_percentage(edge_res)
            benchmarks.append({
                "name": name,
                "latency_ms": round(elapsed, 2),
                "edge_density_pct": round(pct, 2),
                "resolution": f"{edge_res.shape[1]}x{edge_res.shape[0]}"
            })
            images_dict[name] = img_to_base64(edge_res)

        # Histogram data for Otsu
        hist, bin_edges = np.histogram(gray, bins=64, range=(0, 256), density=True)
        otsu_val, _ = compute_otsu_threshold(gray)

        return {
            "status": "success",
            "original_image": original_b64,
            "gray_image": gray_b64,
            "benchmarks": benchmarks,
            "edge_images": images_dict,
            "otsu_threshold": round(otsu_val, 1),
            "histogram": {
                "density": hist.tolist(),
                "bins": bin_edges.tolist()
            }
        }

    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.exit(0)
        data = json.loads(raw_input)
        out = handle_request(data)
        print(json.dumps(out))
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)
