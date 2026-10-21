"""
Edge Detection Tool: Research-Based Streamlit Dashboard
=======================================================
Comparative Analysis and Research-Based Implementation of Canny, Sobel,
Laplacian, and LoG Edge Detection Operators with Otsu-Canny Automation.

Reference Paper:
Sun, L., & Shakhor, S. (2024). Performance Investigating of Several Classical
Edge Detection Operator for Segmentation. Advances in Transdisciplinary Engineering,
Vol. 50, pp. 497-506. DOI: https://doi.org/10.3233/ATDE240501
"""

import io
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import core modular algorithms
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

# Page configuration
st.set_page_config(
    page_title="EDGEFORM // Visual Laboratory for Edge Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #101113;
        color: #F2F1ED;
    }
    .main-header {
        font-family: monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #F2F1ED;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: monospace;
        font-size: 0.95rem;
        color: #91949A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #141619;
        border: 1px solid #2C2E32;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-verified {
        background-color: #1e291e;
        color: #C7F36B;
        border: 1px solid #2f4523;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: monospace;
    }
    .badge-approx {
        background-color: #2e261a;
        color: #f7b731;
        border: 1px solid #4d3d23;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper function to generate synthetic images if no file is uploaded
def create_synthetic_sample(sample_type="Geometric Shapes"):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    if sample_type == "Geometric Shapes":
        # Dark slate background
        img[:] = (35, 30, 30)
        # White rectangle
        cv2.rectangle(img, (40, 40), (140, 140), (240, 240, 240), -1)
        # Cyan circle
        cv2.circle(img, (200, 90), 50, (230, 210, 50), -1)
        # Yellow triangle
        pts = np.array([[70, 260], [150, 180], [230, 260]], np.int32)
        cv2.fillPoly(img, [pts], (50, 220, 240))
        # High contrast boundary line
        cv2.line(img, (10, 280), (290, 280), (255, 255, 255), 2)
    elif sample_type == "High Contrast Circle":
        img[:] = (20, 20, 20)
        cv2.circle(img, (150, 150), 90, (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 45, (80, 80, 80), -1)
    elif sample_type == "Textured Gradient":
        x = np.linspace(0, 255, 300, dtype=np.uint8)
        grad = np.tile(x, (300, 1))
        img[:, :, 0] = grad
        img[:, :, 1] = grad
        img[:, :, 2] = grad
        cv2.circle(img, (150, 150), 60, (255, 255, 255), 3)
        cv2.rectangle(img, (40, 40), (100, 100), (0, 0, 0), 3)
    return img


# Helper to convert numpy array to download bytes
def to_image_bytes(arr: np.ndarray, format: str = "PNG") -> bytes:
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


# Sidebar: Image selection and source configuration
st.sidebar.markdown("### 📁 Image Source")
source_option = st.sidebar.radio("Input Source", ["Sample Images (Built-in)", "Upload Your Own Image"])

current_image = None
image_name = "sample.png"

if source_option == "Sample Images (Built-in)":
    sample_choice = st.sidebar.selectbox("Choose Sample Image", ["Geometric Shapes", "High Contrast Circle", "Textured Gradient"])
    current_image = create_synthetic_sample(sample_choice)
    image_name = f"{sample_choice.lower().replace(' ', '_')}.png"
else:
    uploaded_file = st.sidebar.file_uploader("Upload Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        decoded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if decoded is not None:
            # Convert OpenCV BGR to RGB for consistent Streamlit handling
            current_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            image_name = uploaded_file.name
        else:
            st.sidebar.error("Error decoding image file. Please upload a valid image.")
    else:
        st.sidebar.info("Upload an image above or switch to Sample Images.")
        current_image = create_synthetic_sample("Geometric Shapes")
        image_name = "geometric_shapes.png"

# App Header
st.markdown("<div class='main-header'>EDGEFORM // See the structure within.</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>A visual laboratory for exploring image gradients, edges, and structural information through classical computer vision algorithms · Sun & Shakhor (2024, DOI: 10.3233/ATDE240501)</div>",
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Overview",
    "🔬 Edge Detection",
    "📄 Research Paper (Sun & Shakhor 2024)",
    "📊 Experiments & Evaluation",
    "🎓 Viva Preparation & About"
])

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tab1:
    st.markdown("### Project Purpose & Theoretical Foundations")
    st.write(
        """
        Edge detection is a foundational task in image segmentation and computer vision. Edges correspond to
        significant local discontinuities in image brightness, typically indicating object boundaries,
        surface markings, or depth transitions.
        """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Supported Operators & Mathematical Formulation")
        st.markdown(
            """
            1. **Canny Edge Detector (1986)**
               - Multi-stage optimal detector: Smoothing $\\rightarrow$ Gradient Computation $\\rightarrow$ Non-Maximum Suppression $\\rightarrow$ Double Thresholding $\\rightarrow$ Hysteresis Tracking.
               - Produces clean, single-pixel thin edges with minimal false positives.

            2. **Sobel Operator**
               - First-order discrete differential operator computing horizontal ($G_x$) and vertical ($G_y$) gradients.
               - Combined Gradient Magnitude: $G = \\sqrt{G_x^2 + G_y^2}$.
               - Emphasizes edges with steep intensity transitions; requires normalization to prevent overflow.

            3. **Laplacian Operator**
               - Second-order spatial derivative: $\\nabla^2 f = \\frac{\\partial^2 f}{\\partial x^2} + \\frac{\\partial^2 f}{\\partial y^2}$.
               - Detects edge boundaries via zero-crossings; extremely sensitive to high-frequency noise.

            4. **Laplacian of Gaussian (LoG)**
               - Pre-filters image with Gaussian kernel to suppress noise, then applies the Laplacian operator.
               - Isolates true zero-crossings corresponding to object boundaries at defined scales.

            5. **Prewitt & Roberts Operators**
               - Lightweight first-order spatial difference operators used as comparative baselines.
            """
        )

    with col2:
        st.markdown("#### Processing Pipeline & Research Workflow")
        st.markdown(
            """
            ```mermaid
            graph TD
                A[Input Image RGB/Grayscale] --> B[Ensure Grayscale uint8]
                B --> C1[Canny: Blur + Gx/Gy + NMS + Hysteresis]
                B --> C2[Sobel: CV_64F Gx, Gy + Magnitude + Normalization]
                B --> C3[Laplacian: Second-Order Spatial Derivative]
                B --> C4[LoG: Gaussian Smoothing + Laplacian Zero-Crossing]
                B --> C5[Research Otsu-Canny: Adaptive Histogram Inter-Class Variance]
                C1 --> D[Evaluation & Benchmarking]
                C2 --> D
                C3 --> D
                C4 --> D
                C5 --> D
                D --> E[Execution Latency ms & Edge Density %]
            ```
            """
        )
        st.info(
            "💡 **Research Focus**: Addressing the critical limitation of manual Canny thresholding "
            "using Otsu's maximum inter-class variance method (Sun & Shakhor, 2024)."
        )

# ==========================================
# TAB 2: EDGE DETECTION INTERACTIVE LAB
# ==========================================
with tab2:
    st.markdown("### Interactive Edge Detection Studio")
    st.write("Tune algorithm parameters, inspect intermediate representations, and download processed outputs.")

    control_col, display_col = st.columns([1, 2])

    with control_col:
        st.markdown("#### ⚙️ Algorithm & Parameters")
        algorithm = st.selectbox(
            "Select Algorithm",
            ["Canny", "Sobel", "Laplacian", "Laplacian of Gaussian (LoG)", "Prewitt", "Roberts Cross"]
        )

        params = {}
        if algorithm == "Canny":
            params["apply_blur"] = st.checkbox("Apply Gaussian Pre-filter", value=True)
            params["kernel_size"] = st.slider("Gaussian Kernel Size", min_value=3, max_value=15, value=5, step=2)
            params["sigma"] = st.slider("Gaussian Sigma", min_value=0.5, max_value=4.0, value=1.4, step=0.1)
            params["low_threshold"] = st.slider("Lower Hysteresis Threshold (T_low)", 0.0, 255.0, 50.0, 1.0)
            params["high_threshold"] = st.slider("Upper Hysteresis Threshold (T_high)", 0.0, 255.0, 150.0, 1.0)
            params["l2_gradient"] = st.checkbox("Use L2 Norm sqrt(Gx^2 + Gy^2)", value=False)

        elif algorithm == "Sobel":
            params["kernel_size"] = st.selectbox("Sobel Kernel Aperture (ksize)", [1, 3, 5, 7], index=1)
            params["normalize"] = st.checkbox("Normalize Magnitude to [0, 255]", value=True)
            apply_thresh = st.checkbox("Apply Binarization Threshold", value=False)
            params["threshold"] = st.slider("Binary Threshold Cut-off", 0.0, 255.0, 60.0) if apply_thresh else None

        elif algorithm == "Laplacian":
            params["kernel_size"] = st.selectbox("Laplacian Aperture Size", [1, 3, 5, 7], index=1)
            params["pre_blur"] = st.checkbox("Apply Gaussian Pre-smoothing", value=False)
            params["blur_kernel_size"] = st.slider("Smoothing Kernel Size", 3, 15, 3, step=2) if params["pre_blur"] else 3
            apply_thresh = st.checkbox("Apply Binarization Threshold", value=False)
            params["threshold"] = st.slider("Laplacian Response Cut-off", 0.0, 255.0, 40.0) if apply_thresh else None

        elif algorithm == "Laplacian of Gaussian (LoG)":
            params["gaussian_kernel_size"] = st.slider("Gaussian Kernel Size", 3, 19, 5, step=2)
            params["sigma"] = st.slider("Gaussian Sigma", 0.3, 4.0, 1.0, step=0.1)
            params["laplacian_kernel_size"] = st.selectbox("Laplacian Kernel Size", [1, 3, 5], index=1)
            params["zero_crossing"] = st.checkbox("Detect Zero-Crossings (True LoG)", value=True)
            if params["zero_crossing"]:
                params["zero_crossing_threshold"] = st.slider("Zero-Crossing Sensitivity Threshold", 0.1, 10.0, 2.0, 0.1)

        elif algorithm in ["Prewitt", "Roberts Cross"]:
            apply_thresh = st.checkbox("Apply Binarization Threshold", value=False)
            params["threshold"] = st.slider("Binary Threshold Cut-off", 0.0, 255.0, 50.0) if apply_thresh else None

    # Run processing
    if current_image is not None:
        gray_image = ensure_grayscale(current_image)

        # Measure real execution time
        if algorithm == "Canny":
            res, exec_time = measure_execution_time(canny_edge_detection, current_image, **params)
            edge_map = res["edges"]
        elif algorithm == "Sobel":
            res, exec_time = measure_execution_time(sobel_edge_detection, current_image, **params)
            edge_map = res["edges"]
        elif algorithm == "Laplacian":
            res, exec_time = measure_execution_time(laplacian_edge_detection, current_image, **params)
            edge_map = res["edges"]
        elif algorithm == "Laplacian of Gaussian (LoG)":
            res, exec_time = measure_execution_time(laplacian_of_gaussian, current_image, **params)
            edge_map = res["edges"]
        elif algorithm == "Prewitt":
            res, exec_time = measure_execution_time(prewitt_edge_detection, current_image, **params)
            edge_map = res["edges"]
        elif algorithm == "Roberts Cross":
            res, exec_time = measure_execution_time(roberts_cross_edge_detection, current_image, **params)
            edge_map = res["edges"]

        edge_pct = calculate_edge_pixel_percentage(edge_map)

        with display_col:
            st.markdown("#### 🖼️ Visual Comparison")
            c1, c2 = st.columns(2)
            with c1:
                st.image(current_image, caption=f"Original Input ({current_image.shape[1]}x{current_image.shape[0]})", use_container_width=True)
            with c2:
                st.image(edge_map, caption=f"{algorithm} Output Edge Map", use_container_width=True, clamp=True)

            # Metrics
            st.markdown("#### ⏱️ Real Execution Diagnostics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Execution Latency", f"{exec_time:.2f} ms")
            with m2:
                st.metric("Edge Pixel Density", f"{edge_pct:.2f} %")
            with m3:
                st.metric("Output Resolution", f"{edge_map.shape[1]} × {edge_map.shape[0]}")

            st.caption(
                "⚠️ **Metric Note**: Edge Pixel Density indicates edge density/sparsity. "
                "Higher percentage does not indicate higher accuracy. Execution latency is wall-clock runtime in ms."
            )

            # Download button
            download_bytes = to_image_bytes(edge_map)
            st.download_button(
                label=f"⬇️ Download {algorithm} Edge Map (PNG)",
                data=download_bytes,
                file_name=f"{algorithm.lower().replace(' ', '_')}_edges.png",
                mime="image/png"
            )

            # Extra visual components for Sobel
            if algorithm == "Sobel" and "grad_x" in res:
                st.markdown("##### Intermediate Directional Gradients")
                g1, g2 = st.columns(2)
                with g1:
                    st.image(res["grad_x"], caption="Horizontal Gradient |Gx| (Vertical Edges)", use_container_width=True)
                with g2:
                    st.image(res["grad_y"], caption="Vertical Gradient |Gy| (Horizontal Edges)", use_container_width=True)

# ==========================================
# TAB 3: RESEARCH PAPER IMPLEMENTATION
# ==========================================
with tab3:
    st.markdown("### Research Paper Implementation & Comparative Analysis")
    st.markdown(
        """
        **Reference Publication:**  
        *Lei Sun and Shariar Shakhor (2024)*.  
        **"Performance Investigating of Several Classical Edge Detection Operator for Segmentation"**  
        *Advances in Transdisciplinary Engineering*, Vol. 50, pp. 497–506.  
        **DOI:** [https://doi.org/10.3233/ATDE240501](https://doi.org/10.3233/ATDE240501)
        """
    )

    st.markdown("---")

    # Research Methodology Disclosure
    st.markdown("#### 1. Research Objectives & Methodology Summary")
    st.markdown(
        """
        - **Core Problem Addressed:** Classical Canny edge detection depends heavily on two user-configured hysteresis thresholds
          ($T_{low}, T_{high}$). In real-world segmentation pipelines, manual threshold setting is subjective, fragile to illumination changes,
          and poorly suited for automated computer vision systems.
        - **Proposed Approach:** Investigating the incorporation of **Otsu's thresholding method** to automatically derive
          optimal hysteresis boundaries based on image statistical distribution.
        - **Threshold Formulation:**
          - Otsu computes threshold $T_{otsu}$ by maximizing inter-class variance $\\sigma_B^2(t) = \\omega_0(t)\\omega_1(t)(\\mu_0(t) - \\mu_1(t))^2$.
          - Upper hysteresis threshold: $T_{high} = T_{otsu}$.
          - Lower hysteresis threshold: $T_{low} = \\alpha \\times T_{otsu}$, where $\\alpha \\in [0.4, 0.6]$ (typically $\\alpha = 0.5$).
        """
    )

    # Verification vs Approximation Status Badge
    st.markdown(
        """
        <div style="background-color: #F1F5F9; border-left: 4px solid #0EA5E9; padding: 12px; border-radius: 4px; margin: 15px 0;">
            <strong>Research Provenance & Integrity Status:</strong><br>
            <span class='badge-verified'>VERIFIED PRINCIPLE</span> The theoretical framework of utilizing Otsu's inter-class variance for Canny automation is verified from the paper.<br>
            <span class='badge-approx'>RESEARCH-INSPIRED APPROXIMATION</span> Because exact conference hyperparameter tables for proprietary datasets are commercial proceedings, this implementation is explicitly labelled <strong>Otsu-Canny Variant (Paper-Inspired)</strong> with configurable ratio &alpha;.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 2. Manual Canny vs. Otsu-Adaptive Canny Comparison")

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.markdown("##### Adaptive Tuning")
        otsu_ratio = st.slider("Hysteresis Scaling Ratio (α = T_low / T_high)", 0.2, 0.8, 0.5, 0.05,
                               help="Ratio between lower and upper threshold derived from Otsu.")
        smooth_sigma = st.slider("Pre-blur Sigma", 0.5, 3.0, 1.4, 0.1)

        manual_low = st.slider("Manual T_low for Baseline", 0.0, 255.0, 50.0)
        manual_high = st.slider("Manual T_high for Baseline", 0.0, 255.0, 150.0)

    # Compute executions
    gray_target = ensure_grayscale(current_image)
    otsu_val, _ = compute_otsu_threshold(gray_target)

    # Manual Canny
    manual_res, t_manual = measure_execution_time(
        canny_edge_detection, current_image,
        low_threshold=manual_low, high_threshold=manual_high, sigma=smooth_sigma
    )

    # Paper-Inspired Otsu-Canny
    otsu_res, t_otsu = measure_execution_time(
        otsu_canny_paper_inspired, current_image,
        ratio=otsu_ratio, sigma=smooth_sigma
    )

    # Gradient-Domain Extension
    grad_otsu_res, t_grad_otsu = measure_execution_time(
        gradient_otsu_canny, current_image,
        ratio=otsu_ratio, sigma=smooth_sigma
    )

    with rc2:
        st.markdown(
            f"**Calculated Otsu Threshold ($T_{{otsu}}$):** `{otsu_res['otsu_threshold']:.1f}` | "
            f"**Derived $T_{{high}}$:** `{otsu_res['high_threshold']:.1f}` | "
            f"**Derived $T_{{low}}$:** `{otsu_res['low_threshold']:.1f}`"
        )

        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.image(manual_res["edges"], caption=f"Manual Canny (T=[{manual_low:.0f}, {manual_high:.0f}])", use_container_width=True)
            st.caption(f"Time: {t_manual:.2f} ms | Edge Pixels: {calculate_edge_pixel_percentage(manual_res['edges']):.2f}%")
        with cmp2:
            st.image(otsu_res["edges"], caption=f"Otsu-Canny Variant (Paper-Inspired) (T=[{otsu_res['low_threshold']:.0f}, {otsu_res['high_threshold']:.0f}])", use_container_width=True)
            st.caption(f"Time: {t_otsu:.2f} ms | Edge Pixels: {calculate_edge_pixel_percentage(otsu_res['edges']):.2f}%")

    st.markdown("---")
    st.markdown("#### 3. Otsu Threshold Histogram Placement")
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.hist(gray_target.ravel(), bins=64, range=(0, 256), color="#3B82F6", alpha=0.6, density=True)
    ax.axvline(otsu_res["high_threshold"], color="#EF4444", linestyle="--", linewidth=2, label=f"T_high (Otsu: {otsu_res['high_threshold']:.1f})")
    ax.axvline(otsu_res["low_threshold"], color="#F59E0B", linestyle=":", linewidth=2, label=f"T_low (α={otsu_ratio}: {otsu_res['low_threshold']:.1f})")
    ax.set_title("Grayscale Intensity Histogram with Otsu Hysteresis Thresholds", fontsize=11)
    ax.set_xlabel("Pixel Intensity [0 - 255]")
    ax.set_ylabel("Normalized Density")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown("#### 4. Critical Discussion: Assumptions & Limitations")
    st.markdown(
        """
        - **Bimodal Histogram Assumption:** Otsu's method fundamentally assumes the intensity distribution is bimodal (distinct foreground and background classes).
          When an image exhibits uniform gradients, extreme illumination changes, or complex textures, the global intensity Otsu threshold may land in a valley that does not correspond to edge salience.
        - **Intensity Space vs. Gradient Space:** Standard Otsu operates on pixel intensities, whereas Canny's hysteresis thresholds operate on spatial gradient magnitudes $\\nabla I$.
          Our research extension **Gradient-Domain Otsu-Canny** bridges this gap by calculating Otsu directly over the Sobel gradient magnitude distribution.
        """
    )

# ==========================================
# TAB 4: EXPERIMENTS AND EVALUATION
# ==========================================
with tab4:
    st.markdown("### Benchmarking & Comparative Evaluation Suite")
    st.write("Standardized comparative execution across all core algorithms on the identical input image.")

    # Run all algorithms
    bench_results = []

    methods = [
        ("Canny (Manual Default)", lambda img: canny_edge_detection(img, low_threshold=50, high_threshold=150)["edges"]),
        ("Otsu-Canny (Paper-Inspired)", lambda img: otsu_canny_paper_inspired(img, ratio=0.5)["edges"]),
        ("Sobel Magnitude", lambda img: sobel_edge_detection(img, kernel_size=3)["edges"]),
        ("Laplacian", lambda img: laplacian_edge_detection(img, kernel_size=3, pre_blur=False)["edges"]),
        ("Laplacian of Gaussian (LoG)", lambda img: laplacian_of_gaussian(img, gaussian_kernel_size=5, sigma=1.0, zero_crossing=True)["edges"]),
        ("Prewitt", lambda img: prewitt_edge_detection(img)["edges"]),
        ("Roberts Cross", lambda img: roberts_cross_edge_detection(img)["edges"])
    ]

    edge_maps = {}
    for name, func in methods:
        edge_out, exec_ms = measure_execution_time(func, current_image)
        pct = calculate_edge_pixel_percentage(edge_out)
        bench_results.append({
            "Algorithm": name,
            "Execution Time (ms)": round(exec_ms, 2),
            "Edge Pixel Density (%)": round(pct, 2),
            "Output Dimensions": f"{edge_out.shape[1]}x{edge_out.shape[0]}",
            "Data Type": str(edge_out.dtype)
        })
        edge_maps[name] = edge_out

    df_bench = pd.DataFrame(bench_results)

    # 4-Column Grid Display of Outputs
    st.markdown("#### Side-by-Side Filter Outputs")
    row1_cols = st.columns(4)
    row1_cols[0].image(current_image, caption="Original Input", use_container_width=True)
    row1_cols[1].image(edge_maps["Canny (Manual Default)"], caption="Canny (Manual)", use_container_width=True, clamp=True)
    row1_cols[2].image(edge_maps["Otsu-Canny (Paper-Inspired)"], caption="Otsu-Canny (Paper)", use_container_width=True, clamp=True)
    row1_cols[3].image(edge_maps["Sobel Magnitude"], caption="Sobel Magnitude", use_container_width=True, clamp=True)

    row2_cols = st.columns(4)
    row2_cols[0].image(edge_maps["Laplacian"], caption="Laplacian (Noisy)", use_container_width=True, clamp=True)
    row2_cols[1].image(edge_maps["Laplacian of Gaussian (LoG)"], caption="LoG (Zero-Crossing)", use_container_width=True, clamp=True)
    row2_cols[2].image(edge_maps["Prewitt"], caption="Prewitt", use_container_width=True, clamp=True)
    row2_cols[3].image(edge_maps["Roberts Cross"], caption="Roberts Cross", use_container_width=True, clamp=True)

    st.markdown("#### Benchmark Results Table")
    st.dataframe(df_bench, use_container_width=True)

    # Download CSV button
    csv_bytes = df_bench.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Benchmark Results to CSV",
        data=csv_bytes,
        file_name="edge_detection_benchmark_results.csv",
        mime="text/csv"
    )

    # Bar chart for execution times
    st.markdown("#### Execution Latency Comparison (ms)")
    chart_data = df_bench.set_index("Algorithm")[["Execution Time (ms)"]]
    st.bar_chart(chart_data)

    st.markdown("---")
    st.markdown("#### 🧪 Controlled Noise Degradation Experiment")
    st.write("Evaluate how each algorithm performs when controlled Gaussian or Salt-and-Pepper noise corrupts the image.")

    noise_col1, noise_col2 = st.columns([1, 2])
    with noise_col1:
        noise_type = st.radio("Noise Type", ["Gaussian Noise", "Salt & Pepper Noise"])
        if noise_type == "Gaussian Noise":
            noise_sigma = st.slider("Noise Sigma (Standard Deviation)", 5.0, 60.0, 25.0)
            noisy_test_img = add_gaussian_noise(current_image, sigma=noise_sigma, seed=42)
        else:
            noise_amt = st.slider("Noise Amount (% corrupted)", 0.01, 0.20, 0.05, 0.01)
            noisy_test_img = add_salt_and_pepper_noise(current_image, amount=noise_amt, seed=42)

    with noise_col2:
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            st.image(noisy_test_img, caption=f"Corrupted Input ({noise_type})", use_container_width=True)
        with nc2:
            res_canny_noise = canny_edge_detection(noisy_test_img)["edges"]
            st.image(res_canny_noise, caption=f"Canny (Filtered): {calculate_edge_pixel_percentage(res_canny_noise):.1f}% edges", use_container_width=True)
        with nc3:
            res_lap_noise = laplacian_edge_detection(noisy_test_img)["edges"]
            st.image(res_lap_noise, caption=f"Laplacian (Sensitive): {calculate_edge_pixel_percentage(res_lap_noise):.1f}% edges", use_container_width=True)

    st.info(
        "🧠 **Observation for Viva:** Notice that the unblurred Laplacian suffers severe catastrophic breakdown "
        "under noise because the second-order derivative amplifies high-frequency variations. "
        "In contrast, Canny and LoG incorporate Gaussian smoothing, suppressing spurious noise spikes."
    )

# ==========================================
# TAB 5: ABOUT AND VIVA PREPARATION
# ==========================================
with tab5:
    st.markdown("### Technical Mentorship & Viva Voce Preparation")
    st.write(
        """
        Comprehensive question-and-answer guide designed for college project demonstrations,
        oral examinations (viva), and technical interviews.
        """
    )

    viva_qa = [
        {
            "q": "1. What is the fundamental difference between first-order and second-order edge detection operators?",
            "a": "First-order operators (Sobel, Prewitt, Roberts) compute the first spatial derivative (gradient vector) of image intensity. Edges correspond to local maxima in the gradient magnitude. Second-order operators (Laplacian, LoG) compute the second derivative of intensity; edges correspond to zero-crossings where the second derivative changes sign."
        },
        {
            "q": "2. Why does the Laplacian operator produce double edges and amplify noise?",
            "a": "The Laplacian operator computes the sum of second derivatives (d^2f/dx^2 + d^2f/dy^2). Differentiating twice acts as a high-pass filter that severely multiplies high-frequency components, which includes sensor noise. Furthermore, an intensity step edge produces a positive peak followed by a negative peak in the second derivative, leading to a zero-crossing between two extreme values (visible as double lines if absolute response is thresholded)."
        },
        {
            "q": "3. Explain each of the five stages of Canny's edge detection algorithm.",
            "a": "1. Gaussian Smoothing: Convolves the image with a 2D Gaussian kernel to reduce noise.\n2. Gradient Calculation: Applies Sobel kernels to obtain horizontal (Gx) and vertical (Gy) gradients, followed by magnitude G and orientation theta = arctan(Gy / Gx).\n3. Non-Maximum Suppression (NMS): Discretizes gradient directions into 0, 45, 90, 135 degrees and suppresses any pixel that is not a local maximum along its gradient direction, thinning edges to 1 pixel wide.\n4. Double Thresholding: Uses two thresholds (T_low, T_high) to partition pixels into strong edges (>= T_high), weak edges (T_low <= G < T_high), and non-edges (< T_low).\n5. Edge Tracking by Hysteresis: Preserves weak edges if and only if they are connected (8-connected) to a strong edge; otherwise discards them."
        },
        {
            "q": "4. How does Otsu's thresholding method work mathematically?",
            "a": "Otsu's method assumes the image histogram is bimodal. It iterates through all possible gray-level thresholds t from 0 to 255 to find the threshold t* that maximizes the between-class variance sigma_B^2(t) = omega_0(t) * omega_1(t) * (mu_0(t) - mu_1(t))^2, where omega_0 and omega_1 are the probabilities of the two classes (background and foreground), and mu_0, mu_1 are their respective mean intensities. Maximizing between-class variance is mathematically equivalent to minimizing within-class variance."
        },
        {
            "q": "5. What is the research contribution of Sun & Shakhor (2024)?",
            "a": "Sun & Shakhor (2024) investigated the comparative performance of classical operators (Sobel, Prewitt, Roberts, Laplacian, Canny) for image segmentation and addressed Canny's critical vulnerability: sensitivity to manually configured hysteresis thresholds. They examined using Otsu's thresholding to compute data-driven, adaptive thresholds directly from image statistics, eliminating subjective trial-and-error."
        },
        {
            "q": "6. Why is Sobel magnitude calculated in 64-bit float (CV_64F) instead of 8-bit unsigned integer (uint8)?",
            "a": "Standard 8-bit unsigned integers (uint8) range from 0 to 255. Image gradient differences can be negative (e.g., 0 - 255 = -255), and squared magnitudes can easily exceed 255. If computed in uint8, negative numbers cause underflow wrap-around (e.g. -1 becomes 255) and large values saturate at 255, destroying gradient fidelity. Using CV_64F prevents arithmetic overflow/underflow, after which values are normalized and converted to uint8."
        },
        {
            "q": "7. Why is LoG (Laplacian of Gaussian) distinct from plain Laplacian?",
            "a": "Plain Laplacian has zero noise attenuation; computing second derivatives on raw pixel noise generates widespread spurious artifacts. LoG explicitly smooths the image with a Gaussian filter of scale sigma first (or analytically convolves with the LoG kernel), suppressing high-frequency noise before computing the second derivative, making zero-crossing detection physically meaningful."
        },
        {
            "q": "8. Why does a higher 'Edge Pixel Percentage' NOT mean higher accuracy?",
            "a": "Edge pixel percentage measures edge density, not correctness. A noisy detector or an unblurred Laplacian will mark large patches of noise as edge pixels, producing an inflated edge percentage (e.g., 30-40%) despite failing completely at boundary segmentation. Conversely, an optimal Canny detector yields thin, precise 1-pixel-wide contours with low edge pixel percentage (e.g., 2-5%) and high semantic accuracy."
        },
        {
            "q": "9. What are the key limitations of the Otsu-Canny method?",
            "a": "Otsu relies on the assumption of a bimodal intensity distribution. In images with unimodal distributions, continuous intensity gradients, or severe non-uniform illumination, Otsu's global threshold may split the histogram at an unnatural point. Furthermore, standard Otsu operates on pixel intensities, whereas Canny's hysteresis acts on spatial gradient magnitudes."
        },
        {
            "q": "10. How do Prewitt and Roberts operators compare to Sobel?",
            "a": "Sobel uses kernel weights [1, 2, 1] which provides smoothing perpendicular to the derivative direction, offering better noise resistance than Prewitt's uniform [1, 1, 1] weights. Roberts uses a compact 2x2 diagonal difference kernel, making it computationally fast with high spatial resolution, but more sensitive to noise due to its small support size."
        }
    ]

    for item in viva_qa:
        with st.expander(item["q"], expanded=False):
            st.markdown(item["a"])

    st.markdown("---")
    st.markdown("#### System & Environment Information")
    st.markdown(
        f"""
        - **OpenCV Version:** `{cv2.__version__}`
        - **NumPy Version:** `{np.__version__}`
        - **Execution Platform:** Python 3.10+ / Linux Environment
        - **College Project:** Comparative Analysis of Classical Edge Detection Operators
        """
    )
