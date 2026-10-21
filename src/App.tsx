import React, { useState, useEffect, useRef } from 'react';
import {
  Sliders,
  BarChart3,
  GraduationCap,
  Download,
  Upload,
  RefreshCw,
  Info,
  ShieldCheck,
  AlertTriangle,
  Zap,
  Activity,
  ChevronDown,
  ChevronRight,
  Eye,
  FileText,
  SplitSquareVertical,
  Columns2
} from 'lucide-react';
import LandingHero from './components/LandingHero';

interface ProcessResponse {
  status: string;
  algorithm: string;
  original_image: string;
  gray_image: string;
  edge_image: string;
  latency_ms: number;
  edge_density_pct: number;
  width: number;
  height: number;
  grad_x_image?: string;
  grad_y_image?: string;
  extra?: Record<string, any>;
  message?: string;
}

interface BenchmarkItem {
  name: string;
  latency_ms: number;
  edge_density_pct: number;
  resolution: string;
}

interface BenchmarkResponse {
  status: string;
  original_image: string;
  gray_image: string;
  benchmarks: BenchmarkItem[];
  edge_images: Record<string, string>;
  otsu_threshold: number;
  histogram: {
    density: number[];
    bins: number[];
  };
}

export default function App() {
  // Navigation: Studio (default), Research, Benchmarks, Viva, Overview
  const [activeTab, setActiveTab] = useState<'studio' | 'research' | 'benchmarks' | 'viva' | 'overview'>('studio');

  // Input Image State
  const [customImageBase64, setCustomImageBase64] = useState<string | null>(null);
  const [imageFileName, setImageFileName] = useState<string>('');
  const [imagePreview, setImagePreview] = useState<string>('');
  const [grayPreview, setGrayPreview] = useState<string>('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [viewSourceMode, setViewSourceMode] = useState<'color' | 'grayscale'>('color');

  // Studio Display Mode: 'side-by-side' | 'split' | 'edge-only'
  const [displayMode, setDisplayMode] = useState<'side-by-side' | 'split' | 'edge-only'>('side-by-side');
  const [splitSliderPos, setSplitSliderPos] = useState<number>(50);

  // Algorithm Selection
  const [selectedAlgo, setSelectedAlgo] = useState<string>('Canny');

  // Algorithm Parameters
  const [cannyParams, setCannyParams] = useState({
    low_threshold: 50,
    high_threshold: 150,
    kernel_size: 5,
    sigma: 1.4,
    apply_blur: true,
    l2_gradient: false
  });

  const [sobelParams, setSobelParams] = useState({
    kernel_size: 3,
    sobel_mode: 'combined' as 'combined' | 'horizontal' | 'vertical',
    normalize: true,
    apply_threshold: false,
    threshold: 60
  });

  const [laplacianParams, setLaplacianParams] = useState({
    kernel_size: 3,
    pre_blur: false,
    blur_kernel_size: 3,
    apply_threshold: false,
    threshold: 40
  });

  const [logParams, setLogParams] = useState({
    gaussian_kernel_size: 5,
    sigma: 1.0,
    laplacian_kernel_size: 3,
    zero_crossing: true,
    zero_crossing_threshold: 2.0
  });

  const [prewittParams, setPrewittParams] = useState({
    apply_threshold: false,
    threshold: 50
  });

  const [robertsParams, setRobertsParams] = useState({
    apply_threshold: false,
    threshold: 50
  });

  // Research State (Sun & Shakhor 2024)
  const [otsuRatio, setOtsuRatio] = useState<number>(0.5);
  const [otsuSigma, setOtsuSigma] = useState<number>(1.4);
  const [researchData, setResearchData] = useState<{
    manual?: ProcessResponse;
    otsu?: ProcessResponse;
  }>({});
  const [researchLoading, setResearchLoading] = useState(false);

  // Benchmarks State
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkResponse | null>(null);
  const [benchLoading, setBenchLoading] = useState(false);
  const [noiseType, setNoiseType] = useState<'none' | 'gaussian' | 'salt_and_pepper'>('none');
  const [noiseSigma, setNoiseSigma] = useState<number>(25.0);
  const [noiseAmount, setNoiseAmount] = useState<number>(0.05);

  // Active Process Result
  const [currentResult, setCurrentResult] = useState<ProcessResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Viva Accordion State
  const [openVivaIndex, setOpenVivaIndex] = useState<number | null>(0);

  // File Upload Reference
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Run processing whenever algorithm, parameters, or uploaded image change
  useEffect(() => {
    if (customImageBase64) {
      runDetection();
    } else {
      setCurrentResult(null);
      setImagePreview('');
      setGrayPreview('');
    }
  }, [
    selectedAlgo,
    cannyParams,
    sobelParams,
    laplacianParams,
    logParams,
    prewittParams,
    robertsParams,
    customImageBase64
  ]);

  useEffect(() => {
    setResearchData({});
    setBenchmarkData(null);
  }, [customImageBase64]);

  const getPayloadParams = () => {
    switch (selectedAlgo) {
      case 'Canny':
        return cannyParams;
      case 'Sobel':
        return sobelParams;
      case 'Laplacian':
        return laplacianParams;
      case 'LoG':
        return logParams;
      case 'Prewitt':
        return prewittParams;
      case 'Roberts':
        return robertsParams;
      default:
        return {};
    }
  };

  const runDetection = async () => {
    if (!customImageBase64) {
      setCurrentResult(null);
      setImagePreview('');
      setGrayPreview('');
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);
    try {
      const payload: any = {
        action: 'process',
        algorithm: selectedAlgo,
        params: getPayloadParams(),
        image_base64: customImageBase64,
      };

      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.message || `Server error: ${res.status}`);
      }

      const data: ProcessResponse = await res.json();
      setCurrentResult(data);
      setImagePreview(data.original_image);
      setGrayPreview(data.gray_image);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to process image');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    setErrorMessage(null);
    const file = e.target.files?.[0];
    if (!file) return;

    const validMimes = ['image/png', 'image/jpeg', 'image/jpg'];
    const validExtensions = ['.png', '.jpg', '.jpeg'];
    const lowerName = file.name.toLowerCase();
    const hasValidExt = validExtensions.some((ext) => lowerName.endsWith(ext));
    const hasValidMime = validMimes.includes(file.type);

    if (!hasValidMime && !hasValidExt) {
      setUploadError('Unsupported file type. Please upload a PNG, JPG, or JPEG image.');
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onerror = () => {
      setUploadError('Error reading file. Please select another image.');
      if (fileInputRef.current) fileInputRef.current.value = '';
    };
    reader.onload = (event) => {
      const b64 = event.target?.result as string;
      if (!b64 || !b64.startsWith('data:image/')) {
        setUploadError('Invalid image data. Please upload a valid PNG, JPG, or JPEG image.');
        if (fileInputRef.current) fileInputRef.current.value = '';
        return;
      }
      setImageFileName(file.name);
      setCustomImageBase64(b64);
      setUploadError(null);
    };
    reader.readAsDataURL(file);
  };

  const runResearchComparison = async () => {
    if (!customImageBase64) return;
    setResearchLoading(true);
    try {
      const basePayload: any = { image_base64: customImageBase64 };

      const resManual = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...basePayload,
          action: 'process',
          algorithm: 'Canny',
          params: { low_threshold: 50, high_threshold: 150, sigma: otsuSigma },
        }),
      });
      const dataManual = await resManual.json();

      const resOtsu = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...basePayload,
          action: 'process',
          algorithm: 'Otsu-Canny',
          params: { ratio: otsuRatio, sigma: otsuSigma },
        }),
      });
      const dataOtsu = await resOtsu.json();

      setResearchData({
        manual: dataManual,
        otsu: dataOtsu,
      });
    } catch (err) {
      console.error(err);
    } finally {
      setResearchLoading(false);
    }
  };

  const runBenchmark = async () => {
    if (!customImageBase64) return;
    setBenchLoading(true);
    try {
      const payload: any = {
        action: 'benchmark',
        image_base64: customImageBase64,
        noise_type: noiseType,
        noise_sigma: noiseSigma,
        noise_amount: noiseAmount,
      };

      const res = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data: BenchmarkResponse = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setBenchLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'research' && customImageBase64 && !researchData.otsu) {
      runResearchComparison();
    }
    if (activeTab === 'benchmarks' && customImageBase64 && !benchmarkData) {
      runBenchmark();
    }
  }, [activeTab, customImageBase64]);

  const exportBenchmarkCSV = () => {
    if (!benchmarkData) return;
    const headers = ['Algorithm', 'Execution Time (ms)', 'Edge Pixel Density (%)', 'Resolution'];
    const rows = benchmarkData.benchmarks.map((b) => [
      `"${b.name}"`,
      b.latency_ms,
      b.edge_density_pct,
      `"${b.resolution}"`,
    ]);
    const csvContent =
      'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'edge_detection_benchmark_results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const vivaQuestions = [
    {
      q: '1. What is the fundamental difference between first-order and second-order edge detection operators?',
      a: 'First-order operators (Sobel, Prewitt, Roberts) evaluate the spatial gradient vector of image intensity. Edges correspond to local maxima (peaks) of the gradient magnitude. Second-order operators (Laplacian, LoG) evaluate the second spatial derivative of intensity; edges are identified at zero-crossings where the second derivative changes sign across inflection points.',
    },
    {
      q: '2. Why does the Laplacian operator produce double edges and heavily amplify noise?',
      a: 'The Laplacian computes the sum of second derivatives (d²f/dx² + d²f/dy²). Differentiating an image twice acts as a high-pass frequency filter that squares the spectral frequency response, multiplying high-frequency noise spikes. Additionally, a step intensity edge causes a positive overshoot on the dark side and a negative undershoot on the bright side. In an absolute response |∇²f|, both positive and negative peaks are positive, displaying as two parallel lines around the true edge.',
    },
    {
      q: '3. Explain each of the five stages of Canny edge detection.',
      a: '1. Gaussian Smoothing: Convolves the image with a 2D Gaussian kernel to suppress noise.\n2. Gradient Calculation: Applies Sobel kernels to obtain horizontal (Gx) and vertical (Gy) gradients, magnitude G = √(Gx² + Gy²), and angle θ = arctan(Gy/Gx).\n3. Non-Maximum Suppression (NMS): Discretizes gradient directions into 0°, 45°, 90°, 135° and suppresses any pixel that is not a local maximum along its gradient normal, thinning edges to 1 pixel wide.\n4. Double Thresholding: Uses two thresholds (T_low, T_high) to partition pixels into strong (≥ T_high), weak (T_low ≤ G < T_high), and non-edges (< T_low).\n5. Edge Tracking by Hysteresis: Preserves weak edges if and only if they are connected through an 8-neighborhood path to a strong edge.',
    },
    {
      q: '4. How does Otsu’s thresholding method work mathematically?',
      a: 'Otsu’s method searches for the threshold t* that maximizes between-class variance: σ_B²(t) = ω₀(t)·ω₁(t)·[μ₀(t) - μ₁(t)]², where ω₀, ω₁ are class probabilities for background and foreground, and μ₀, μ₁ are class mean intensities. Maximizing between-class variance is mathematically equivalent to minimizing the weighted within-class variance σ_W²(t) = ω₀σ₀² + ω₁σ₁².',
    },
    {
      q: '5. What is the research contribution of Sun & Shakhor (2024)?',
      a: 'Sun & Shakhor (2024) [DOI: 10.3233/ATDE240501] conducted a systematic performance investigation of classical operators for segmentation. They addressed Canny’s sensitivity to manual dual thresholds (T_low, T_high) by investigating the use of Otsu’s thresholding to compute data-driven, adaptive thresholds directly from image statistics.',
    },
    {
      q: '6. Why must Sobel gradients be computed using 64-bit float (CV_64F) rather than 8-bit unsigned integer (uint8)?',
      a: 'Pixel gradients can be negative (e.g., 0 - 255 = -255), and squared magnitudes easily exceed 255. In uint8, negative values wrap around to 255 (underflow), and large values saturate at 255. Computing in CV_64F prevents numerical loss and arithmetic overflow before normalizing back to [0, 255].',
    },
    {
      q: '7. Why is LoG (Laplacian of Gaussian) distinct from plain Laplacian?',
      a: 'Plain Laplacian applies second derivatives directly to raw pixels without noise filtering. LoG explicitly smooths the image first with a Gaussian kernel of scale σ, regularizing the differentiation and making zero-crossing detection robust against sensor noise.',
    },
    {
      q: '8. Why does a higher "Edge Pixel Percentage" NOT mean higher accuracy?',
      a: 'Edge pixel percentage measures edge density, not correctness. An unblurred Laplacian on a noisy image flags random noise spikes as edges, yielding high percentages (e.g. 35%) with almost zero semantic validity. A clean, well-tuned Canny detector yields thin, precise single-pixel contours with low density (e.g. 2–5%) and high ground-truth accuracy.',
    },
    {
      q: '9. What are the key limitations of the Otsu-Canny method?',
      a: 'Otsu assumes a bimodal intensity distribution. When an image has a unimodal histogram, continuous illumination gradients, or complex multi-modal textures, Otsu may place the threshold in an unnatural location. Furthermore, standard Otsu operates on pixel intensities, whereas Canny hysteresis acts on gradient magnitudes.',
    },
    {
      q: '10. Compare Sobel, Prewitt, and Roberts operators.',
      a: 'Sobel uses weights [-1, 0, 1] with orthogonal smoothing [1, 2, 1], giving superior noise suppression. Prewitt uses uniform weights [1, 1, 1], making it computationally simpler but slightly noisier. Roberts uses 2×2 diagonal differences, offering minimal computation and high spatial localization, but high sensitivity to noise.',
    },
  ];

  const scrollToLab = () => {
    const el = document.getElementById('lab-workspace');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-[#101113] text-[#F2F1ED] flex flex-col font-sans selection:bg-[#C7F36B] selection:text-[#101113]">
      {/* Immersive Landing Section (Full Screen, Zero Top Header / Nav) */}
      <LandingHero onExplore={scrollToLab} />

      {/* Main Lab Application Workspace */}
      <div id="lab-workspace" className="flex flex-col min-h-screen">
        {/* Single Unified Header */}
        <header className="border-b border-[#2C2E32] sticky top-0 z-30 bg-[#101113]/95 backdrop-blur-md px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-bold tracking-wider text-[#F2F1ED]">
              EDGEFORM
            </span>
            <span className="text-[11px] text-[#91949A] font-mono hidden sm:inline">
              / Edge Detection Tool
            </span>
          </div>

          {/* Unified Primary Navigation */}
          <nav className="flex items-center gap-1.5 text-xs font-mono">
            <button
              onClick={() => setActiveTab('studio')}
              className={`px-3 py-1.5 rounded transition-colors ${
                activeTab === 'studio'
                  ? 'bg-[#1e2024] text-[#C7F36B] font-semibold border border-[#2C2E32]'
                  : 'text-[#91949A] hover:text-[#F2F1ED]'
              }`}
            >
              Studio
            </button>
            <button
              onClick={() => setActiveTab('research')}
              className={`px-3 py-1.5 rounded transition-colors ${
                activeTab === 'research'
                  ? 'bg-[#1e2024] text-[#C7F36B] font-semibold border border-[#2C2E32]'
                  : 'text-[#91949A] hover:text-[#F2F1ED]'
              }`}
            >
              Research (Otsu)
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`px-3 py-1.5 rounded transition-colors ${
                activeTab === 'benchmarks'
                  ? 'bg-[#1e2024] text-[#C7F36B] font-semibold border border-[#2C2E32]'
                  : 'text-[#91949A] hover:text-[#F2F1ED]'
              }`}
            >
              Benchmarks
            </button>
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-3 py-1.5 rounded transition-colors ${
                activeTab === 'overview'
                  ? 'bg-[#1e2024] text-[#C7F36B] font-semibold border border-[#2C2E32]'
                  : 'text-[#91949A] hover:text-[#F2F1ED]'
              }`}
            >
              Overview
            </button>
          </nav>
        </div>

        {/* Current Status */}
        <div className="flex items-center gap-3 text-xs font-mono text-[#91949A]">
          {currentResult && (
            <span className="hidden sm:inline">
              {currentResult.width}×{currentResult.height} px
            </span>
          )}
          <span className="text-[#C7F36B] text-[11px] font-bold">
            {isProcessing ? 'PROCESSING...' : 'READY'}
          </span>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">

        {/* ========================================================= */}
        {/* TAB 1: STUDIO (PRIMARY WORKSPACE) */}
        {/* ========================================================= */}
        {activeTab === 'studio' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Controls Panel (4 cols) */}
            <div className="lg:col-span-4 space-y-4">
              {/* Image Input Section */}
              <div className="bg-[#141619] rounded-lg p-4 border border-[#2C2E32] space-y-3">
                <div className="flex items-center justify-between text-xs font-mono font-semibold text-[#F2F1ED]">
                  <span>1. Input Image</span>
                  {customImageBase64 && (
                    <button
                      onClick={() => {
                        setCustomImageBase64(null);
                        setImageFileName('');
                        setImagePreview('');
                        setGrayPreview('');
                        setCurrentResult(null);
                        setUploadError(null);
                        if (fileInputRef.current) fileInputRef.current.value = '';
                      }}
                      className="text-[#91949A] hover:text-red-400 text-[11px] transition-colors"
                    >
                      Clear
                    </button>
                  )}
                </div>

                {/* Upload Control */}
                <div>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept="image/png,image/jpeg,image/jpg"
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full py-2.5 px-3 border border-[#2C2E32] hover:border-[#C7F36B] bg-[#101113] text-[#F2F1ED] hover:text-[#C7F36B] rounded text-xs font-mono font-semibold transition-colors flex items-center justify-center gap-2"
                  >
                    <Upload className="w-3.5 h-3.5 text-[#C7F36B]" />
                    <span>Upload Image</span>
                  </button>
                </div>

                {/* Validation Error Message */}
                {uploadError && (
                  <div className="p-2.5 bg-red-950/40 border border-red-500/40 text-red-300 rounded text-xs font-mono flex items-center gap-2">
                    <AlertTriangle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                    <span>{uploadError}</span>
                  </div>
                )}

                {/* Preview of the uploaded image or empty state */}
                {customImageBase64 ? (
                  <div className="space-y-1.5">
                    <div className="w-full h-36 bg-black rounded border border-[#2C2E32] overflow-hidden flex items-center justify-center">
                      <img
                        src={customImageBase64}
                        alt="Uploaded preview"
                        className="max-h-full max-w-full object-contain"
                      />
                    </div>
                    {imageFileName && (
                      <div className="text-[10px] font-mono text-[#91949A] truncate">
                        {imageFileName}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-4 border border-dashed border-[#2C2E32] rounded text-center text-[#91949A] text-xs font-mono">
                    Upload an image to get started.
                  </div>
                )}
              </div>

              {/* Algorithm Selection & Dynamic Tuning */}
              <div className="bg-[#141619] rounded-lg p-4 border border-[#2C2E32] space-y-4">
                <div className="flex items-center justify-between text-xs font-mono font-semibold text-[#F2F1ED]">
                  <span>2. Algorithm</span>
                  <button
                    onClick={runDetection}
                    disabled={isProcessing}
                    className="text-[#91949A] hover:text-[#C7F36B] flex items-center gap-1 text-[11px]"
                    title="Re-run processing"
                  >
                    <RefreshCw className={`w-3 h-3 ${isProcessing ? 'animate-spin' : ''}`} /> Refresh
                  </button>
                </div>

                <div>
                  <select
                    value={selectedAlgo}
                    onChange={(e) => setSelectedAlgo(e.target.value)}
                    className="w-full text-xs bg-[#101113] border border-[#2C2E32] rounded p-2 font-mono text-[#C7F36B] font-bold focus:border-[#C7F36B] focus:outline-hidden"
                  >
                    <option value="Canny">Canny (Multi-Stage Optimal)</option>
                    <option value="Sobel">Sobel (First-Order Gradient)</option>
                    <option value="Laplacian">Laplacian (Second-Order)</option>
                    <option value="LoG">Laplacian of Gaussian (LoG)</option>
                    <option value="Prewitt">Prewitt (Difference Operator)</option>
                    <option value="Roberts">Roberts Cross (Diagonal 2×2)</option>
                  </select>
                </div>

                {/* Relevant Parameters Only */}
                <div className="pt-2 border-t border-[#2C2E32] space-y-3 text-xs font-mono">
                  {selectedAlgo === 'Canny' && (
                    <>
                      <div>
                        <div className="flex justify-between text-[#91949A] mb-1">
                          <span>Lower Threshold (T_low)</span>
                          <span className="text-[#C7F36B] font-bold">{cannyParams.low_threshold}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="255"
                          value={cannyParams.low_threshold}
                          onChange={(e) =>
                            setCannyParams({ ...cannyParams, low_threshold: Number(e.target.value) })
                          }
                          className="w-full accent-[#C7F36B]"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[#91949A] mb-1">
                          <span>Upper Threshold (T_high)</span>
                          <span className="text-[#C7F36B] font-bold">{cannyParams.high_threshold}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="255"
                          value={cannyParams.high_threshold}
                          onChange={(e) =>
                            setCannyParams({ ...cannyParams, high_threshold: Number(e.target.value) })
                          }
                          className="w-full accent-[#C7F36B]"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[#91949A] mb-1">
                          <span>Gaussian Blur Kernel</span>
                          <span className="text-[#C7F36B]">{cannyParams.kernel_size}×{cannyParams.kernel_size}</span>
                        </div>
                        <select
                          value={cannyParams.kernel_size}
                          onChange={(e) =>
                            setCannyParams({ ...cannyParams, kernel_size: Number(e.target.value) })
                          }
                          className="w-full text-xs bg-[#101113] border border-[#2C2E32] rounded p-1.5 text-[#F2F1ED]"
                        >
                          <option value="3">3 × 3</option>
                          <option value="5">5 × 5 (Standard)</option>
                          <option value="7">7 × 7</option>
                        </select>
                      </div>

                      <div className="flex items-center gap-2 pt-1">
                        <input
                          type="checkbox"
                          id="apply_blur"
                          checked={cannyParams.apply_blur}
                          onChange={(e) => setCannyParams({ ...cannyParams, apply_blur: e.target.checked })}
                          className="rounded text-[#C7F36B] bg-[#101113] border-[#2C2E32]"
                        />
                        <label htmlFor="apply_blur" className="text-[#F2F1ED] cursor-pointer">
                          Gaussian noise pre-filter
                        </label>
                      </div>
                    </>
                  )}

                  {selectedAlgo === 'Sobel' && (
                    <>
                      <div>
                        <label className="text-[#91949A] block mb-1.5">Gradient Component</label>
                        <div className="grid grid-cols-3 gap-1 bg-[#101113] p-1 border border-[#2C2E32] rounded">
                          <button
                            onClick={() => setSobelParams({ ...sobelParams, sobel_mode: 'combined' })}
                            className={`py-1 text-[11px] rounded ${
                              sobelParams.sobel_mode === 'combined'
                                ? 'bg-[#1c1e22] text-[#C7F36B] font-bold border border-[#2C2E32]'
                                : 'text-[#91949A] hover:text-[#F2F1ED]'
                            }`}
                          >
                            Magnitude |G|
                          </button>
                          <button
                            onClick={() => setSobelParams({ ...sobelParams, sobel_mode: 'horizontal' })}
                            className={`py-1 text-[11px] rounded ${
                              sobelParams.sobel_mode === 'horizontal'
                                ? 'bg-[#1c1e22] text-[#C7F36B] font-bold border border-[#2C2E32]'
                                : 'text-[#91949A] hover:text-[#F2F1ED]'
                            }`}
                          >
                            Horiz Gx
                          </button>
                          <button
                            onClick={() => setSobelParams({ ...sobelParams, sobel_mode: 'vertical' })}
                            className={`py-1 text-[11px] rounded ${
                              sobelParams.sobel_mode === 'vertical'
                                ? 'bg-[#1c1e22] text-[#C7F36B] font-bold border border-[#2C2E32]'
                                : 'text-[#91949A] hover:text-[#F2F1ED]'
                            }`}
                          >
                            Vert Gy
                          </button>
                        </div>
                      </div>

                      <div>
                        <label className="text-[#91949A] block mb-1">Aperture Size (ksize)</label>
                        <select
                          value={sobelParams.kernel_size}
                          onChange={(e) =>
                            setSobelParams({ ...sobelParams, kernel_size: Number(e.target.value) })
                          }
                          className="w-full text-xs bg-[#101113] border border-[#2C2E32] rounded p-1.5 text-[#F2F1ED]"
                        >
                          <option value="1">1 (1st Difference)</option>
                          <option value="3">3 × 3 (Standard)</option>
                          <option value="5">5 × 5</option>
                          <option value="7">7 × 7</option>
                        </select>
                      </div>

                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="sobel_thresh"
                          checked={sobelParams.apply_threshold}
                          onChange={(e) =>
                            setSobelParams({ ...sobelParams, apply_threshold: e.target.checked })
                          }
                          className="rounded text-[#C7F36B] bg-[#101113] border-[#2C2E32]"
                        />
                        <label htmlFor="sobel_thresh" className="text-[#F2F1ED] cursor-pointer">
                          Binarize threshold
                        </label>
                      </div>

                      {sobelParams.apply_threshold && (
                        <div>
                          <div className="flex justify-between text-[#91949A] mb-1">
                            <span>Cut-off Value</span>
                            <span className="text-[#C7F36B]">{sobelParams.threshold}</span>
                          </div>
                          <input
                            type="range"
                            min="0"
                            max="255"
                            value={sobelParams.threshold}
                            onChange={(e) =>
                              setSobelParams({ ...sobelParams, threshold: Number(e.target.value) })
                            }
                            className="w-full accent-[#C7F36B]"
                          />
                        </div>
                      )}
                    </>
                  )}

                  {selectedAlgo === 'Laplacian' && (
                    <>
                      <div>
                        <label className="text-[#91949A] block mb-1">Aperture Size</label>
                        <select
                          value={laplacianParams.kernel_size}
                          onChange={(e) =>
                            setLaplacianParams({ ...laplacianParams, kernel_size: Number(e.target.value) })
                          }
                          className="w-full text-xs bg-[#101113] border border-[#2C2E32] rounded p-1.5 text-[#F2F1ED]"
                        >
                          <option value="1">1 (4-Neighborhood)</option>
                          <option value="3">3 × 3</option>
                          <option value="5">5 × 5</option>
                        </select>
                      </div>

                      <div className="flex items-center gap-2 pt-1">
                        <input
                          type="checkbox"
                          id="lap_preblur"
                          checked={laplacianParams.pre_blur}
                          onChange={(e) =>
                            setLaplacianParams({ ...laplacianParams, pre_blur: e.target.checked })
                          }
                          className="rounded text-[#C7F36B] bg-[#101113] border-[#2C2E32]"
                        />
                        <label htmlFor="lap_preblur" className="text-[#F2F1ED] cursor-pointer">
                          Gaussian pre-smoothing
                        </label>
                      </div>
                    </>
                  )}

                  {selectedAlgo === 'LoG' && (
                    <>
                      <div>
                        <div className="flex justify-between text-[#91949A] mb-1">
                          <span>Gaussian Scale (σ)</span>
                          <span className="text-[#C7F36B]">{logParams.sigma}</span>
                        </div>
                        <input
                          type="range"
                          min="0.4"
                          max="3.0"
                          step="0.1"
                          value={logParams.sigma}
                          onChange={(e) => setLogParams({ ...logParams, sigma: Number(e.target.value) })}
                          className="w-full accent-[#C7F36B]"
                        />
                      </div>

                      <div>
                        <label className="text-[#91949A] block mb-1">Gaussian Filter Size</label>
                        <select
                          value={logParams.gaussian_kernel_size}
                          onChange={(e) =>
                            setLogParams({ ...logParams, gaussian_kernel_size: Number(e.target.value) })
                          }
                          className="w-full text-xs bg-[#101113] border border-[#2C2E32] rounded p-1.5 text-[#F2F1ED]"
                        >
                          <option value="3">3 × 3</option>
                          <option value="5">5 × 5</option>
                          <option value="7">7 × 7</option>
                        </select>
                      </div>

                      <div className="flex items-center gap-2 pt-1">
                        <input
                          type="checkbox"
                          id="log_zc"
                          checked={logParams.zero_crossing}
                          onChange={(e) => setLogParams({ ...logParams, zero_crossing: e.target.checked })}
                          className="rounded text-[#C7F36B] bg-[#101113] border-[#2C2E32]"
                        />
                        <label htmlFor="log_zc" className="text-[#F2F1ED] cursor-pointer">
                          Zero-crossing detection
                        </label>
                      </div>
                    </>
                  )}

                  {(selectedAlgo === 'Prewitt' || selectedAlgo === 'Roberts') && (
                    <div>
                      <label className="text-[#91949A] block mb-1">Optional Binarization Threshold</label>
                      <input
                        type="range"
                        min="0"
                        max="255"
                        value={selectedAlgo === 'Prewitt' ? prewittParams.threshold : robertsParams.threshold}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          if (selectedAlgo === 'Prewitt') setPrewittParams({ apply_threshold: true, threshold: val });
                          else setRobertsParams({ apply_threshold: true, threshold: val });
                        }}
                        className="w-full accent-[#C7F36B]"
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* Actions & Metrics */}
              {currentResult && (
                <div className="bg-[#141619] rounded-lg p-4 border border-[#2C2E32] space-y-3 font-mono text-xs">
                  <div className="grid grid-cols-2 gap-3 text-center">
                    <div className="bg-[#101113] p-2.5 rounded border border-[#2C2E32]">
                      <div className="text-[10px] text-[#91949A]">LATENCY</div>
                      <div className="text-base font-bold text-[#F2F1ED] mt-0.5">
                        {currentResult.latency_ms} <span className="text-[10px] font-normal text-[#91949A]">ms</span>
                      </div>
                    </div>
                    <div className="bg-[#101113] p-2.5 rounded border border-[#2C2E32]">
                      <div className="text-[10px] text-[#91949A]">EDGE DENSITY</div>
                      <div className="text-base font-bold text-[#F2F1ED] mt-0.5">
                        {currentResult.edge_density_pct} <span className="text-[10px] font-normal text-[#91949A]">%</span>
                      </div>
                    </div>
                  </div>

                  <a
                    href={currentResult.edge_image}
                    download={`${selectedAlgo.toLowerCase()}_edge_output.png`}
                    className="w-full py-2 bg-[#C7F36B] hover:bg-[#d6fa8c] text-[#101113] rounded font-bold tracking-wider uppercase transition-colors flex items-center justify-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" /> Download Edge PNG
                  </a>
                </div>
              )}
            </div>

            {/* Right Visual Canvas (8 cols) */}
            <div className="lg:col-span-8 space-y-3">
              {errorMessage && (
                <div className="p-3 bg-red-950/40 border border-red-500/40 text-red-200 rounded text-xs font-mono flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  <span>{errorMessage}</span>
                </div>
              )}

              {/* Canvas Controls Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono pb-1 border-b border-[#2C2E32]">
                <div className="flex items-center gap-2">
                  <span className="text-[#91949A]">View Mode:</span>
                  <div className="inline-flex bg-[#141619] border border-[#2C2E32] rounded p-0.5">
                    <button
                      onClick={() => setDisplayMode('side-by-side')}
                      className={`px-2.5 py-1 rounded flex items-center gap-1 ${
                        displayMode === 'side-by-side' ? 'bg-[#1c1e22] text-[#C7F36B] font-bold' : 'text-[#91949A] hover:text-[#F2F1ED]'
                      }`}
                    >
                      <Columns2 className="w-3 h-3" /> Side-by-Side
                    </button>
                    <button
                      onClick={() => setDisplayMode('split')}
                      className={`px-2.5 py-1 rounded flex items-center gap-1 ${
                        displayMode === 'split' ? 'bg-[#1c1e22] text-[#C7F36B] font-bold' : 'text-[#91949A] hover:text-[#F2F1ED]'
                      }`}
                    >
                      <SplitSquareVertical className="w-3 h-3" /> Split Slider
                    </button>
                    <button
                      onClick={() => setDisplayMode('edge-only')}
                      className={`px-2.5 py-1 rounded flex items-center gap-1 ${
                        displayMode === 'edge-only' ? 'bg-[#1c1e22] text-[#C7F36B] font-bold' : 'text-[#91949A] hover:text-[#F2F1ED]'
                      }`}
                    >
                      <Eye className="w-3 h-3" /> Edge Only
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-[#91949A]">Source:</span>
                  <button
                    onClick={() => setViewSourceMode(viewSourceMode === 'color' ? 'grayscale' : 'color')}
                    className="px-2 py-0.5 border border-[#2C2E32] hover:border-[#91949A] rounded text-[11px] text-[#F2F1ED]"
                  >
                    {viewSourceMode === 'color' ? 'RGB Color' : 'Grayscale'}
                  </button>
                </div>
              </div>

              {/* Main Visual Panels */}
              {displayMode === 'side-by-side' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Left: Input Image */}
                  <div className="bg-[#141619] rounded-lg p-3 border border-[#2C2E32] flex flex-col">
                    <div className="flex items-center justify-between text-xs font-mono text-[#91949A] mb-2">
                      <span>Source ({viewSourceMode === 'color' ? 'RGB' : 'Grayscale'})</span>
                      <span>{currentResult ? `${currentResult.width}×${currentResult.height}` : ''}</span>
                    </div>
                    <div className="w-full h-80 sm:h-96 bg-black rounded flex items-center justify-center overflow-hidden border border-[#2C2E32]">
                      {imagePreview ? (
                        <img
                          src={viewSourceMode === 'color' ? imagePreview : grayPreview}
                          alt="Input"
                          className="max-h-full max-w-full object-contain"
                        />
                      ) : (
                        <span className="text-[#91949A] text-xs font-mono">Upload an image to get started.</span>
                      )}
                    </div>
                  </div>

                  {/* Right: Edge Result */}
                  <div className="bg-[#141619] rounded-lg p-3 border border-[#2C2E32] flex flex-col">
                    <div className="flex items-center justify-between text-xs font-mono text-[#91949A] mb-2">
                      <span className="text-[#C7F36B] font-bold">
                        {selectedAlgo} {selectedAlgo === 'Sobel' ? `(${sobelParams.sobel_mode})` : 'Response'}
                      </span>
                      <span>{currentResult ? `${currentResult.latency_ms} ms` : ''}</span>
                    </div>
                    <div className="w-full h-80 sm:h-96 bg-black rounded flex items-center justify-center overflow-hidden border border-[#2C2E32]">
                      {currentResult?.edge_image ? (
                        <img
                          src={currentResult.edge_image}
                          alt="Edge response"
                          className="max-h-full max-w-full object-contain"
                        />
                      ) : (
                        <span className="text-[#91949A] text-xs font-mono">
                          {isProcessing ? 'Computing edge response...' : 'Upload an image to get started.'}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Split Comparison Slider Mode */}
              {displayMode === 'split' && (
                <div className="bg-[#141619] rounded-lg p-3 border border-[#2C2E32] space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono text-[#91949A]">
                    <span>Drag slider or inspect spatial transition</span>
                    <span>Split: {splitSliderPos}%</span>
                  </div>

                  {imagePreview ? (
                    <div
                      className="relative w-full h-96 bg-black rounded overflow-hidden select-none cursor-ew-resize border border-[#2C2E32]"
                      onMouseMove={(e) => {
                        if (e.buttons === 1) {
                          const rect = e.currentTarget.getBoundingClientRect();
                          const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                          setSplitSliderPos(Math.round((x / rect.width) * 100));
                        }
                      }}
                      onClick={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                        setSplitSliderPos(Math.round((x / rect.width) * 100));
                      }}
                    >
                      {/* Background: Original Image */}
                      <img
                        src={viewSourceMode === 'color' ? imagePreview : grayPreview}
                        alt="Source"
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                      />

                      {/* Foreground: Edge Response clipped */}
                      {currentResult?.edge_image && (
                        <div
                          className="absolute inset-0 overflow-hidden pointer-events-none"
                          style={{
                            clipPath: `polygon(0 0, ${splitSliderPos}% 0, ${splitSliderPos}% 100%, 0 100%)`
                          }}
                        >
                          <img
                            src={currentResult.edge_image}
                            alt="Edge"
                            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                          />
                        </div>
                      )}

                      {/* Divider Line */}
                      <div
                        className="absolute top-0 bottom-0 w-0.5 bg-[#C7F36B] pointer-events-none"
                        style={{ left: `${splitSliderPos}%` }}
                      />
                    </div>
                  ) : (
                    <div className="w-full h-96 bg-black rounded flex items-center justify-center border border-[#2C2E32] text-[#91949A] text-xs font-mono">
                      Upload an image to get started.
                    </div>
                  )}

                  <div className="flex items-center justify-between text-[11px] font-mono text-[#91949A]">
                    <span>Left: Edge Response</span>
                    <span>Right: Original Source</span>
                  </div>
                </div>
              )}

              {/* Single Edge Only Mode */}
              {displayMode === 'edge-only' && (
                <div className="bg-[#141619] rounded-lg p-3 border border-[#2C2E32]">
                  <div className="flex items-center justify-between text-xs font-mono text-[#91949A] mb-2">
                    <span className="text-[#C7F36B] font-bold">{selectedAlgo} Isolated Edge Map</span>
                    <span>{currentResult ? `${currentResult.width}×${currentResult.height}` : ''}</span>
                  </div>
                  <div className="w-full h-96 sm:h-[480px] bg-black rounded flex items-center justify-center overflow-hidden border border-[#2C2E32]">
                    {currentResult?.edge_image ? (
                      <img
                        src={currentResult.edge_image}
                        alt="Edge Map"
                        className="max-h-full max-w-full object-contain"
                      />
                    ) : (
                      <span className="text-[#91949A] text-xs font-mono">
                        {isProcessing ? 'Computing edge response...' : 'Upload an image to get started.'}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Scientific Disclaimer */}
              <div className="p-3 bg-[#101113] rounded border border-[#2C2E32] text-[11px] font-mono text-[#91949A] leading-relaxed flex items-start gap-2">
                <Info className="w-4 h-4 text-[#C7F36B] shrink-0 mt-0.5" />
                <span>
                  <strong>Diagnostic Note:</strong> Edge Pixel Density represents the proportion of non-zero pixels.
                  It measures boundary sparsity, <em>not mathematical accuracy</em>. Unfiltered operators will yield high
                  density on noisy inputs without true boundary localization.
                </span>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 2: RESEARCH PAPER IMPLEMENTATION */}
        {/* ========================================================= */}
        {activeTab === 'research' && (
          <div className="bg-[#141619] rounded-lg p-5 border border-[#2C2E32] space-y-6">
            {!customImageBase64 ? (
              <div className="py-20 text-center text-[#91949A] font-mono text-xs border border-dashed border-[#2C2E32] rounded">
                Upload an image to get started.
              </div>
            ) : (
              <>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[#2C2E32]">
                  <div>
                    <span className="text-[10px] font-mono text-[#C7F36B] uppercase tracking-wider">
                      Reference: Sun &amp; Shakhor (2024) · DOI: 10.3233/ATDE240501
                    </span>
                    <h2 className="text-xl font-bold text-[#F2F1ED] mt-1">
                      Otsu-Canny Threshold Automation
                    </h2>
                    <p className="text-xs text-[#91949A] font-mono mt-0.5">
                      Automating Canny hysteresis thresholds via maximum inter-class variance
                    </p>
                  </div>

                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="text-[#91949A]">Ratio α (T_low / T_high):</span>
                    <input
                      type="range"
                      min="0.2"
                      max="0.8"
                      step="0.05"
                      value={otsuRatio}
                      onChange={(e) => setOtsuRatio(Number(e.target.value))}
                      className="w-24 accent-[#C7F36B]"
                    />
                    <span className="text-[#C7F36B] font-bold">{otsuRatio}</span>
                    <button
                      onClick={runResearchComparison}
                      disabled={researchLoading}
                      className="px-3 py-1.5 bg-[#C7F36B] text-[#101113] rounded font-bold hover:bg-[#d6fa8c] transition-colors ml-2"
                    >
                      {researchLoading ? 'Evaluating...' : 'Re-Evaluate'}
                    </button>
                  </div>
                </div>

                {/* Side-by-Side Outputs */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-[#101113] rounded p-4 border border-[#2C2E32] space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono text-[#91949A]">
                      <span className="text-[#F2F1ED] font-semibold">1. Manual Canny Baseline (T = [50, 150])</span>
                      <span>{researchData.manual?.latency_ms} ms</span>
                    </div>
                    <div className="h-72 bg-black rounded flex items-center justify-center overflow-hidden border border-[#2C2E32]">
                      {researchData.manual?.edge_image ? (
                        <img src={researchData.manual.edge_image} alt="Manual Canny" className="max-h-full max-w-full object-contain" />
                      ) : (
                        <span className="text-[#91949A] text-xs font-mono">
                          {researchLoading ? 'Computing baseline...' : 'Upload an image to get started.'}
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] font-mono text-[#91949A]">
                      Density: {researchData.manual?.edge_density_pct}%
                    </div>
                  </div>

                  <div className="bg-[#101113] rounded p-4 border border-[#2C2E32] space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono text-[#91949A]">
                      <span className="text-[#C7F36B] font-bold">2. Otsu-Canny Variant (Paper-Inspired)</span>
                      <span className="text-[#C7F36B]">{researchData.otsu?.latency_ms} ms</span>
                    </div>
                    <div className="h-72 bg-black rounded flex items-center justify-center overflow-hidden border border-[#2C2E32]">
                      {researchData.otsu?.edge_image ? (
                        <img src={researchData.otsu.edge_image} alt="Otsu Canny" className="max-h-full max-w-full object-contain" />
                      ) : (
                        <span className="text-[#91949A] text-xs font-mono">
                          {researchLoading ? 'Computing Otsu Canny...' : 'Upload an image to get started.'}
                        </span>
                      )}
                    </div>
                    {researchData.otsu?.extra && (
                      <div className="text-[11px] font-mono text-[#91949A] flex justify-between">
                        <span>T_otsu: {researchData.otsu.extra.otsu_threshold}</span>
                        <span>T_low: {researchData.otsu.extra.low_threshold}</span>
                        <span>T_high: {researchData.otsu.extra.high_threshold}</span>
                        <span>Density: {researchData.otsu.edge_density_pct}%</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Academic Disclosure */}
                <div className="p-4 bg-[#101113] border border-[#2C2E32] rounded text-xs font-mono text-[#91949A] space-y-2">
                  <div className="text-[#C7F36B] font-bold flex items-center gap-1.5 uppercase">
                    <ShieldCheck className="w-4 h-4 text-[#C7F36B]" /> Research Methodology Disclosure
                  </div>
                  <p>
                    Sun &amp; Shakhor (2024) established that manual hysteresis thresholding poses a critical bottleneck for
                    automated segmentation pipelines. Otsu’s discriminant criterion maximizes between-class variance
                    σ_B²(t) = ω₀(t)·ω₁(t)·[μ₀(t) - μ₁(t)]² to compute an objective scalar T_otsu. Because specific empirical scaling
                    tables for proprietary conference datasets are closed-source, this implementation is explicitly labelled
                    <strong> “Otsu-Canny Variant (Paper-Inspired)”</strong>, setting T_high = T_otsu and T_low = α · T_otsu.
                  </p>
                </div>
              </>
            )}
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 3: BENCHMARKS & EXPERIMENTS */}
        {/* ========================================================= */}
        {activeTab === 'benchmarks' && (
          <div className="bg-[#141619] rounded-lg p-5 border border-[#2C2E32] space-y-6">
            {!customImageBase64 ? (
              <div className="py-20 text-center text-[#91949A] font-mono text-xs border border-dashed border-[#2C2E32] rounded">
                Upload an image to get started.
              </div>
            ) : (
              <>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[#2C2E32]">
                  <div>
                    <h2 className="text-xl font-bold text-[#F2F1ED]">
                      Algorithm Benchmarking &amp; Noise Experiments
                    </h2>
                    <p className="text-xs font-mono text-[#91949A] mt-0.5">
                      Comparative performance and noise resistance evaluated simultaneously on the identical input image.
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={runBenchmark}
                      disabled={benchLoading}
                      className="px-3.5 py-1.5 bg-[#C7F36B] text-[#101113] rounded text-xs font-mono font-bold hover:bg-[#d6fa8c] transition-colors flex items-center gap-1.5"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${benchLoading ? 'animate-spin' : ''}`} /> Run Suite
                    </button>
                    <button
                      onClick={exportBenchmarkCSV}
                      disabled={!benchmarkData}
                      className="px-3.5 py-1.5 bg-[#101113] border border-[#2C2E32] hover:border-[#91949A] text-[#F2F1ED] rounded text-xs font-mono transition-colors flex items-center gap-1.5"
                    >
                      <Download className="w-3.5 h-3.5" /> Export CSV
                    </button>
                  </div>
                </div>

                {/* Controlled Noise Parameters */}
                <div className="p-3 bg-[#101113] rounded border border-[#2C2E32] flex flex-wrap items-center gap-4 text-xs font-mono">
                  <span className="text-[#C7F36B] font-bold">Controlled Noise:</span>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="radio"
                      name="bench_noise"
                      checked={noiseType === 'none'}
                      onChange={() => setNoiseType('none')}
                      className="accent-[#C7F36B]"
                    />
                    Clean Image
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="radio"
                      name="bench_noise"
                      checked={noiseType === 'gaussian'}
                      onChange={() => setNoiseType('gaussian')}
                      className="accent-[#C7F36B]"
                    />
                    Gaussian Noise (σ)
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="radio"
                      name="bench_noise"
                      checked={noiseType === 'salt_and_pepper'}
                      onChange={() => setNoiseType('salt_and_pepper')}
                      className="accent-[#C7F36B]"
                    />
                    Salt &amp; Pepper
                  </label>

                  {noiseType === 'gaussian' && (
                    <div className="flex items-center gap-2 pl-3 border-l border-[#2C2E32]">
                      <span>Sigma:</span>
                      <input
                        type="range"
                        min="5"
                        max="50"
                        value={noiseSigma}
                        onChange={(e) => setNoiseSigma(Number(e.target.value))}
                        className="w-20 accent-[#C7F36B]"
                      />
                      <span className="text-[#C7F36B]">{noiseSigma}</span>
                    </div>
                  )}
                </div>

                {/* Real Benchmark Table */}
                {benchmarkData && (
                  <div className="space-y-6">
                    <div className="overflow-x-auto rounded border border-[#2C2E32]">
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="bg-[#101113] text-[#91949A] border-b border-[#2C2E32]">
                          <tr>
                            <th className="py-2.5 px-4">Operator</th>
                            <th className="py-2.5 px-4">Execution Latency (ms)</th>
                            <th className="py-2.5 px-4">Edge Pixel Density (%)</th>
                            <th className="py-2.5 px-4">Resolution</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#2C2E32] text-[#F2F1ED]">
                          {benchmarkData.benchmarks.map((item, idx) => (
                            <tr key={idx} className="hover:bg-[#101113]/50 transition-colors">
                              <td className="py-2.5 px-4 font-bold">{item.name}</td>
                              <td className="py-2.5 px-4 text-[#C7F36B]">{item.latency_ms} ms</td>
                              <td className="py-2.5 px-4 text-[#91949A]">{item.edge_density_pct}%</td>
                              <td className="py-2.5 px-4 text-[#91949A]">{item.resolution}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Multi-Filter Visual Grid */}
                    <div>
                      <h4 className="text-xs font-mono font-bold text-[#F2F1ED] mb-3">
                        Comparative Visual Matrix
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {Object.entries(benchmarkData.edge_images).map(([algoName, b64], idx) => (
                          <div key={idx} className="bg-[#101113] border border-[#2C2E32] rounded p-2 flex flex-col items-center">
                            <div className="w-full text-[10px] font-mono text-[#91949A] mb-1.5 truncate">
                              {algoName}
                            </div>
                            <div className="w-full h-32 flex items-center justify-center overflow-hidden bg-black rounded">
                              <img src={b64} alt={algoName} className="max-h-full max-w-full object-contain" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 4: VIVA VOCE & REPORT */}
        {/* ========================================================= */}
        {activeTab === 'viva' && (
          <div className="bg-[#141619] rounded-lg p-5 border border-[#2C2E32] space-y-6">
            <div>
              <h2 className="text-xl font-bold text-[#F2F1ED]">
                Viva Voce Questions &amp; Technical Defense
              </h2>
              <p className="text-xs font-mono text-[#91949A] mt-0.5">
                Ten critical academic questions prepared for college project demonstration and oral examination.
              </p>
            </div>

            <div className="space-y-2.5">
              {vivaQuestions.map((item, idx) => {
                const isOpen = openVivaIndex === idx;
                return (
                  <div key={idx} className="border border-[#2C2E32] rounded bg-[#101113] overflow-hidden">
                    <button
                      onClick={() => setOpenVivaIndex(isOpen ? null : idx)}
                      className="w-full text-left p-3 flex items-center justify-between text-xs font-mono font-bold text-[#F2F1ED] hover:text-[#C7F36B] transition-colors"
                    >
                      <span>{item.q}</span>
                      {isOpen ? <ChevronDown className="w-4 h-4 text-[#C7F36B]" /> : <ChevronRight className="w-4 h-4" />}
                    </button>
                    {isOpen && (
                      <div className="p-3.5 bg-[#141619] text-xs font-mono text-[#91949A] leading-relaxed whitespace-pre-line border-t border-[#2C2E32]">
                        {item.a}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="pt-4 border-t border-[#2C2E32]">
              <h4 className="text-xs font-mono font-bold text-[#F2F1ED] mb-2">
                Local Execution Guide for Demonstration
              </h4>
              <div className="bg-[#101113] p-3 rounded border border-[#2C2E32] font-mono text-xs text-[#91949A] space-y-1.5">
                <p className="text-[#C7F36B]"># 1. Run local Streamlit dashboard</p>
                <p className="text-[#F2F1ED]">streamlit run app.py</p>
                <p className="text-[#C7F36B] mt-2"># 2. Run automated test suite</p>
                <p className="text-[#F2F1ED]">pytest tests/ -v</p>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 5: OVERVIEW & THEORY */}
        {/* ========================================================= */}
        {activeTab === 'overview' && (
          <div className="bg-[#141619] rounded-lg p-5 border border-[#2C2E32] space-y-6">
            <div>
              <h2 className="text-xl font-bold text-[#F2F1ED]">
                Classical Edge Detection Foundations
              </h2>
              <p className="text-xs font-mono text-[#91949A] mt-0.5">
                Mathematical formulations and engineering trade-offs of differential operators.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-4 bg-[#101113] rounded border border-[#2C2E32] space-y-2">
                <span className="text-[#C7F36B] font-bold">1. First-Order Gradient Operators (Sobel, Prewitt)</span>
                <p className="text-[#91949A] leading-relaxed">
                  Evaluates intensity gradients ∇f = [Gx, Gy]ᵀ. Magnitude G = √(Gx² + Gy²). Sobel incorporates [1, 2, 1]
                  smoothing orthogonal to the derivative direction, providing noise suppression. Computed in 64-bit float (CV_64F)
                  to prevent overflow before normalization.
                </p>
              </div>

              <div className="p-4 bg-[#101113] rounded border border-[#2C2E32] space-y-2">
                <span className="text-[#C7F36B] font-bold">2. Second-Order Operators (Laplacian, LoG)</span>
                <p className="text-[#91949A] leading-relaxed">
                  Evaluates ∇²f = d²f/dx² + d²f/dy². Detects zero-crossings. Second differentiation amplifies high-frequency
                  noise. Marr-Hildreth LoG regularizes the derivative by applying Gaussian smoothing G_σ before the Laplacian.
                </p>
              </div>

              <div className="p-4 bg-[#101113] rounded border border-[#2C2E32] space-y-2">
                <span className="text-[#C7F36B] font-bold">3. Multi-Stage Canny Optimization</span>
                <p className="text-[#91949A] leading-relaxed">
                  Gaussian blur → Sobel gradients → Non-Maximum Suppression (thins edges to 1 pixel) → Double Thresholding
                  (T_low, T_high) → 8-connected hysteresis tracking. Produces continuous, minimal-error boundaries.
                </p>
              </div>

              <div className="p-4 bg-[#101113] rounded border border-[#2C2E32] space-y-2">
                <span className="text-[#C7F36B] font-bold">4. Otsu Variance Maximization</span>
                <p className="text-[#91949A] leading-relaxed">
                  Maximizes between-class variance σ_B²(t) = ω₀(t)·ω₁(t)·[μ₀(t) - μ₁(t)]² from the intensity histogram.
                  Sun &amp; Shakhor (2024) demonstrated that coupling Otsu with Canny automates threshold selection.
                </p>
              </div>
            </div>
          </div>
        )}

      </main>
      </div>
    </div>
  );
}
