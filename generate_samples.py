"""
Utility to generate synthetic benchmark and test images in sample_images/.
"""

import os
import cv2
import numpy as np


def generate_samples(output_dir="sample_images"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Geometric Shapes (Rectangle, Circle, Polygon, Lines)
    img_geom = np.zeros((320, 320, 3), dtype=np.uint8)
    img_geom[:] = (35, 30, 30)  # Slate dark background
    cv2.rectangle(img_geom, (40, 40), (140, 140), (240, 240, 240), -1)  # White square
    cv2.circle(img_geom, (210, 90), 50, (230, 210, 50), -1)  # Cyan circle
    pts = np.array([[70, 280], [150, 190], [230, 280]], np.int32)
    cv2.fillPoly(img_geom, [pts], (50, 220, 240))  # Triangle
    cv2.line(img_geom, (15, 300), (305, 300), (255, 255, 255), 2)  # Boundary
    cv2.imwrite(os.path.join(output_dir, "geometric_shapes.png"), img_geom)

    # 2. Concentric Circles (Smooth curves and gradients)
    img_circ = np.zeros((300, 300), dtype=np.uint8)
    img_circ[:] = 20
    cv2.circle(img_circ, (150, 150), 100, 255, -1)
    cv2.circle(img_circ, (150, 150), 60, 120, -1)
    cv2.circle(img_circ, (150, 150), 25, 20, -1)
    cv2.imwrite(os.path.join(output_dir, "concentric_circles.png"), img_circ)

    # 3. High Contrast Grid Box (Benchmarking ground-truth edges)
    img_box = np.zeros((200, 200), dtype=np.uint8)
    img_box[50:150, 50:150] = 255
    cv2.imwrite(os.path.join(output_dir, "box_step_edge.png"), img_box)

    print(f"Sample images successfully generated in '{output_dir}'.")


if __name__ == "__main__":
    generate_samples()
