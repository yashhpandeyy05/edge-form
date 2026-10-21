import React, { useState, useEffect, useRef } from 'react';

interface LandingHeroProps {
  onExplore: () => void;
}

type ScanPhase = 'pause_start' | 'scanning_forward' | 'pause_end' | 'scanning_backward';

interface ImageDimensions {
  width: number;
  height: number;
  aspectRatio: number;
}

export default function LandingHero({ onExplore }: LandingHeroProps) {
  const [scanPercent, setScanPercent] = useState<number>(0);
  const [phase, setPhase] = useState<ScanPhase>('pause_start');
  const [isUserScrubbing, setIsUserScrubbing] = useState<boolean>(false);
  const [, setImageLoaded] = useState<boolean>(false);
  const [imageDimensions, setImageDimensions] = useState<ImageDimensions>({
    width: 750,
    height: 500,
    aspectRatio: 750 / 500,
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const phaseTimerRef = useRef<number>(0);
  const scrubTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Timing constants (in milliseconds)
  const DURATION_PAUSE_START = 1200;
  const DURATION_FORWARD_SCAN = 3200;
  const DURATION_PAUSE_END = 1600;
  const DURATION_BACKWARD_SCAN = 2600;

  // Handle dynamic image dimensions calculation on load
  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (img.naturalWidth && img.naturalHeight) {
      setImageDimensions({
        width: img.naturalWidth,
        height: img.naturalHeight,
        aspectRatio: img.naturalWidth / img.naturalHeight,
      });
    }
    setImageLoaded(true);
  };

  useEffect(() => {
    let currentPhase: ScanPhase = 'pause_start';
    let progress = 0;
    let timer = 0;

    const animate = (currentTime: number) => {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = currentTime;
      }
      const deltaTime = currentTime - lastTimeRef.current;
      lastTimeRef.current = currentTime;

      // Only run auto-animation if user is not actively scrubbing
      if (!isUserScrubbing) {
        if (currentPhase === 'pause_start') {
          timer += deltaTime;
          progress = 0;
          if (timer >= DURATION_PAUSE_START) {
            timer = 0;
            currentPhase = 'scanning_forward';
            setPhase('scanning_forward');
          }
        } else if (currentPhase === 'scanning_forward') {
          timer += deltaTime;
          const ratio = Math.min(timer / DURATION_FORWARD_SCAN, 1);
          // Smooth sine ease in-out
          progress = (1 - Math.cos(ratio * Math.PI)) / 2 * 100;
          if (timer >= DURATION_FORWARD_SCAN) {
            timer = 0;
            progress = 100;
            currentPhase = 'pause_end';
            setPhase('pause_end');
          }
        } else if (currentPhase === 'pause_end') {
          timer += deltaTime;
          progress = 100;
          if (timer >= DURATION_PAUSE_END) {
            timer = 0;
            currentPhase = 'scanning_backward';
            setPhase('scanning_backward');
          }
        } else if (currentPhase === 'scanning_backward') {
          timer += deltaTime;
          const ratio = Math.min(timer / DURATION_BACKWARD_SCAN, 1);
          // Smooth sine ease from 100 down to 0
          progress = 100 - ((1 - Math.cos(ratio * Math.PI)) / 2 * 100);
          if (timer >= DURATION_BACKWARD_SCAN) {
            timer = 0;
            progress = 0;
            currentPhase = 'pause_start';
            setPhase('pause_start');
          }
        }

        setScanPercent(progress);
        phaseTimerRef.current = timer;
      }

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
      if (scrubTimeoutRef.current) {
        clearTimeout(scrubTimeoutRef.current);
      }
    };
  }, [isUserScrubbing]);

  // Handle interactive scrubbing by mouse or touch
  const handleInteraction = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setScanPercent(percent);
    setIsUserScrubbing(true);

    if (scrubTimeoutRef.current) {
      clearTimeout(scrubTimeoutRef.current);
    }

    // Resume automated loop after 2.5s of no interaction
    scrubTimeoutRef.current = setTimeout(() => {
      setIsUserScrubbing(false);
      lastTimeRef.current = null;
    }, 2500);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.buttons === 1) {
      handleInteraction(e.clientX);
    }
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length > 0) {
      handleInteraction(e.touches[0].clientX);
    }
  };

  // Status text for the precision scientific instrument
  const getPhaseDescription = () => {
    if (isUserScrubbing) return `INTERACTIVE PROBE · ${scanPercent.toFixed(1)}%`;
    switch (phase) {
      case 'pause_start':
        return 'OPTICAL CAPTURE · RAW RGB';
      case 'scanning_forward':
        return `ACTIVE SCAN · CANNY GRADIENT [${scanPercent.toFixed(0)}%]`;
      case 'pause_end':
        return 'STRUCTURAL TOPOLOGY · FULL EXTRACTION';
      case 'scanning_backward':
        return `RETRACTING SCAN BEAM [${scanPercent.toFixed(0)}%]`;
    }
  };

  return (
    <section className="min-h-screen w-full bg-[#101113] text-[#F2F1ED] flex flex-col justify-between items-center relative overflow-hidden px-4 sm:px-6 py-6 sm:py-8 md:py-10 select-none">
      {/* Background subtle technical grid pattern */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-[0.035]"
        style={{
          backgroundImage: `linear-gradient(to right, #ffffff 1px, transparent 1px), linear-gradient(to bottom, #ffffff 1px, transparent 1px)`,
          backgroundSize: '48px 48px'
        }}
      />

      {/* Atmospheric center vignette */}
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse 70% 50% at 50% 40%, rgba(199, 243, 107, 0.02) 0%, rgba(16, 17, 19, 0.8) 70%, #101113 100%)'
        }}
      />

      {/* Top Wordmark - Clean, elegant scientific instrument brand mark */}
      <div className="relative z-10 w-full max-w-5xl flex items-center justify-center pt-2 pb-2">
        <div className="inline-flex items-center gap-2.5">
          <span className="w-1.5 h-1.5 bg-[#C7F36B] rotate-45 shadow-[0_0_8px_rgba(199,243,107,0.7)]" />
          <span className="font-mono text-sm sm:text-base font-bold tracking-[0.32em] text-[#F2F1ED] uppercase select-none">
            EDGEFORM
          </span>
        </div>
      </div>

      {/* Hero Content - Centered */}
      <div className="relative z-10 max-w-4xl w-full flex flex-col items-center text-center my-auto py-2">
        {/* Eyebrow */}
        <div className="inline-flex items-center gap-2 mb-3 sm:mb-4 font-mono text-[11px] sm:text-xs tracking-[0.25em] text-[#C7F36B] uppercase">
          <span className="w-1.5 h-1.5 rounded-full bg-[#C7F36B] animate-pulse" />
          <span>COMPUTER VISION / IMAGE ANALYSIS</span>
        </div>

        {/* Main Headline */}
        <h1 className="text-3xl sm:text-5xl md:text-6xl font-medium tracking-tight text-[#F2F1ED] mb-3 sm:mb-4">
          See beyond the image.
        </h1>

        {/* Supporting text */}
        <p className="text-xs sm:text-sm md:text-base text-[#91949A] max-w-xl mx-auto font-normal leading-relaxed mb-6 sm:mb-8">
          Transform ordinary images into their underlying structure with classical edge detection.
        </p>

        {/* Visual Transformation Frame */}
        <div className="w-full max-w-[760px] mx-auto mb-6 sm:mb-8">
          {/* Scientific Frame Header */}
          <div className="flex items-center justify-between text-[11px] font-mono px-3 py-1.5 bg-[#141619] border border-[#2C2E32] border-b-0 rounded-t-md text-[#91949A]">
            <div className="flex items-center gap-2">
              <span className="text-[#F2F1ED] font-semibold">FACADE SPECIMEN</span>
              <span className="text-[#2C2E32]">/</span>
              <span className="hidden sm:inline font-mono">
                {imageDimensions.width}×{imageDimensions.height} px
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#C7F36B]" />
              <span className="text-[#C7F36B]">{getPhaseDescription()}</span>
            </div>
          </div>

          {/* Interactive Image Display Area - Dynamically Sized to Aspect Ratio */}
          <div
            ref={containerRef}
            onClick={(e) => handleInteraction(e.clientX)}
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
            style={{
              aspectRatio: `${imageDimensions.width} / ${imageDimensions.height}`,
            }}
            className="relative w-full bg-[#0c0d0f] border border-[#2C2E32] rounded-b-md overflow-hidden cursor-ew-resize group shadow-2xl transition-[aspect-ratio] duration-200"
          >
            {/* Base Layer: Original Photograph */}
            <img
              src="/sample_images/architectural_facade.png"
              alt="Raw Architectural Photograph"
              onLoad={handleImageLoad}
              className="absolute inset-0 w-full h-full object-cover block pointer-events-none"
            />

            {/* Revealing Layer: Edge Detected Representation */}
            <div
              className="absolute inset-0 pointer-events-none overflow-hidden"
              style={{
                clipPath: `inset(0 ${Math.max(0, 100 - scanPercent)}% 0 0)`
              }}
            >
              <img
                src="/sample_images/architectural_canny_lime.png"
                alt="Edge Detected Structural Outline"
                className="w-full h-full object-cover block"
              />
            </div>

            {/* Glowing Vertical Scan Line */}
            <div
              className="absolute top-0 bottom-0 pointer-events-none z-20"
              style={{
                left: `${scanPercent}%`,
                transform: 'translateX(-50%)'
              }}
            >
              {/* Scan Line Light Beam (Subtle trail behind the line) */}
              <div 
                className="absolute top-0 bottom-0 right-1/2 w-8 sm:w-16 pointer-events-none"
                style={{
                  background: 'linear-gradient(to left, rgba(199, 243, 107, 0.18), transparent)'
                }}
              />

              {/* Main Razor-thin Scan Line */}
              <div 
                className="w-[1.5px] h-full bg-[#C7F36B]"
                style={{
                  boxShadow: '0 0 10px rgba(199, 243, 107, 0.9), 0 0 2px #C7F36B'
                }}
              />

              {/* Top and Bottom Precision Crosshair Ticks */}
              <div className="absolute top-1 left-1/2 -translate-x-1/2 w-2 h-1 bg-[#C7F36B]" />
              <div className="absolute bottom-1 left-1/2 -translate-x-1/2 w-2 h-1 bg-[#C7F36B]" />

              {/* Coordinate readout near the scan line */}
              <div className="absolute bottom-3 left-3 bg-[#101113]/90 backdrop-blur border border-[#C7F36B]/40 px-1.5 py-0.5 rounded text-[9px] font-mono text-[#C7F36B] whitespace-nowrap shadow-md pointer-events-none hidden sm:block">
                X: {Math.round((scanPercent / 100) * imageDimensions.width)}px · {scanPercent.toFixed(1)}%
              </div>
            </div>

            {/* Bottom-right Corner Scrub Prompt */}
            <div className="absolute bottom-2 right-2 z-10 text-[10px] font-mono text-[#91949A]/80 bg-[#101113]/85 backdrop-blur px-2 py-0.5 rounded border border-[#2C2E32]/70 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
              Click or drag to inspect
            </div>
          </div>
        </div>

        {/* Primary Action Button */}
        <div>
          <button
            onClick={onExplore}
            className="group inline-flex items-center gap-2.5 px-6 py-3 rounded bg-[#16181C] hover:bg-[#1D2026] border border-[#2C2E32] hover:border-[#C7F36B]/60 text-[#F2F1ED] hover:text-[#C7F36B] transition-all text-xs sm:text-sm font-mono font-medium shadow-xl hover:shadow-[#C7F36B]/5 active:translate-y-0.5"
          >
            <span>Explore the Lab</span>
            <span className="text-[#C7F36B] text-base transition-transform group-hover:translate-y-0.5">
              ↓
            </span>
          </button>
        </div>
      </div>

      {/* Minimal Footer Stamp */}
      <div className="relative z-10 w-full max-w-5xl flex items-center justify-between text-[11px] font-mono text-[#91949A]/60 pt-4 border-t border-[#2C2E32]/40">
        <span>EDGEFORM CV INSTRUMENT</span>
        <span>CLASSICAL GRADIENTS · CANNY · SOBEL · LAPLACIAN</span>
      </div>
    </section>
  );
}
