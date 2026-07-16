"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Camera, Loader2, StopCircle, Video, AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import * as ort from 'onnxruntime-web';
import { preprocess, postprocess, Prediction } from "@/utils/detect";

// Set WASM paths for ORT to load dependencies from CDN
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

export default function Home() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>();
  const streamRef = useRef<MediaStream | null>(null);

  // Stats
  const stats = {
    fullripe: predictions.filter((p) => p.class_name.toLowerCase() === "fullripe").length,
    semiripe: predictions.filter((p) => p.class_name.toLowerCase() === "semiripe").length,
    unripe: predictions.filter((p) => p.class_name.toLowerCase() === "unripe").length,
    total: predictions.length,
  };

  useEffect(() => {
    // Load ONNX Model
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        // The file must be in frontend/public/best.onnx
        const sess = await ort.InferenceSession.create('/best.onnx', { executionProviders: ['wasm'] });
        setSession(sess);
        setIsModelLoading(false);
      } catch (e: any) {
        console.error("Failed to load model:", e);
        setError("Failed to load the AI model. Ensure best.onnx is in the public folder.");
        setIsModelLoading(false);
      }
    };
    loadModel();

    return () => {
      stopWebcam();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startWebcam = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 640 } }
      });
      streamRef.current = stream;
      setIsStreaming(true); // Mounts the video element
      
      // Wait a tick for React to render the video element
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            detectFrame();
          };
        }
      }, 50);
      
    } catch (err: any) {
      console.error(err);
      setError("Failed to access webcam. Please grant camera permissions.");
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    setPredictions([]);
  };

  const detectFrame = async () => {
    if (!videoRef.current || !session) return;
    
    const vWidth = videoRef.current.videoWidth;
    const vHeight = videoRef.current.videoHeight;
    
    if (vWidth === 0 || vHeight === 0) {
      requestRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    // Prepare an offscreen canvas specifically for the 640x640 model input
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = 640;
    offscreenCanvas.height = 640;
    const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
    
    if (!offscreenCtx) return;

    // Draw the video frame onto the 640x640 canvas (it will squash it if aspect ratio is different, which YOLO expects)
    offscreenCtx.drawImage(videoRef.current, 0, 0, 640, 640);

    try {
      const tensor = preprocess(offscreenCtx, 640, 640);
      const results = await session.run({ images: tensor }); // 'images' is default input name for YOLOv8
      
      const outputName = session.outputNames[0];
      const outputTensor = results[outputName];

      const rawPredictions = postprocess(outputTensor, 640, 640, 0.25, 0.45);
      
      // We must scale the predictions back from 640x640 to the actual video intrinsic size
      const scaleX = vWidth / 640;
      const scaleY = vHeight / 640;

      const scaledPredictions = rawPredictions.map(p => ({
        ...p,
        box: {
          x1: p.box.x1 * scaleX,
          y1: p.box.y1 * scaleY,
          x2: p.box.x2 * scaleX,
          y2: p.box.y2 * scaleY,
        }
      }));

      setPredictions(scaledPredictions);
    } catch (e) {
      console.error("Inference error:", e);
    }

    // Schedule next frame
    requestRef.current = requestAnimationFrame(detectFrame);
  };

  const getClassColor = (className: string) => {
    switch (className.toLowerCase()) {
      case "fullripe": return "rgba(239, 68, 68, 1)";
      case "semiripe": return "rgba(249, 115, 22, 1)";
      case "unripe": return "rgba(34, 197, 94, 1)";
      default: return "rgba(59, 130, 246, 1)";
    }
  };

  const [displayScale, setDisplayScale] = useState({ x: 1, y: 1 });
  
  const updateScale = useCallback(() => {
    if (videoRef.current) {
      const { videoWidth, videoHeight, clientWidth, clientHeight } = videoRef.current;
      if (videoWidth && videoHeight && clientWidth && clientHeight) {
        setDisplayScale({
          x: clientWidth / videoWidth,
          y: clientHeight / videoHeight,
        });
      }
    }
  }, []);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const observer = new ResizeObserver(() => updateScale());
    observer.observe(v);
    return () => observer.disconnect();
  }, [updateScale, isStreaming]);


  return (
    <main className="min-h-screen bg-slate-50/50 text-slate-900 font-sans selection:bg-slate-200">
      <div className="max-w-6xl mx-auto px-4 py-12 flex flex-col gap-10">
        <header className="text-center space-y-4 max-w-2xl mx-auto mt-8">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border bg-white shadow-sm text-xs font-medium text-slate-500 mb-2">
            <span>By Hanif Budi Kurniawan</span>
          </div>
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-zinc-900">
            Strawberry Ripeness Detector
          </h1>
          <p className="text-base text-slate-500">
            Real-time on-device inference using YOLOv8 ONNX and WebGL.
          </p>
        </header>

        <div className="grid lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Actions & Stats */}
          <div className="lg:col-span-4 space-y-6">
            <Card className="border-0 shadow-xl shadow-slate-200/50 bg-white/70 backdrop-blur-xl overflow-hidden ring-1 ring-slate-200">
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Camera className="w-5 h-5 text-slate-400" />
                  Camera Controls
                </h3>

                <div className="space-y-4">
                  {isModelLoading ? (
                    <div className="flex items-center justify-center p-4 bg-slate-100 rounded-lg text-slate-500">
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Loading AI Model...
                    </div>
                  ) : (
                    <>
                      {!isStreaming ? (
                        <Button
                          className="w-full bg-zinc-900 hover:bg-zinc-800 text-white shadow-sm"
                          size="lg"
                          onClick={startWebcam}
                        >
                          <Video className="mr-2 h-5 w-5" />
                          Start Webcam
                        </Button>
                      ) : (
                        <Button
                          className="w-full bg-rose-600 hover:bg-rose-700 text-white shadow-sm"
                          size="lg"
                          onClick={stopWebcam}
                        >
                          <StopCircle className="mr-2 h-5 w-5" />
                          Stop Webcam
                        </Button>
                      )}
                    </>
                  )}
                </div>

                {error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* Dashboard Stats */}
            <div className="space-y-4 animate-in fade-in duration-500">
              <div className="flex items-center gap-4">
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider px-1 shrink-0">
                  Live Results
                </h3>
                <Separator className="flex-1" />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <Card className="border-0 shadow-sm bg-white/70 backdrop-blur-md">
                  <CardContent className="p-4 flex flex-col justify-center items-center text-center space-y-2">
                    <Badge variant="outline" className="text-rose-500 border-rose-200 bg-rose-50">
                      Fullripe
                    </Badge>
                    <span className="text-4xl font-extrabold text-slate-700">{stats.fullripe}</span>
                  </CardContent>
                </Card>
                <Card className="border-0 shadow-sm bg-white/70 backdrop-blur-md">
                  <CardContent className="p-4 flex flex-col justify-center items-center text-center space-y-2">
                    <Badge variant="outline" className="text-orange-500 border-orange-200 bg-orange-50">
                      Semiripe
                    </Badge>
                    <span className="text-4xl font-extrabold text-slate-700">{stats.semiripe}</span>
                  </CardContent>
                </Card>
                <Card className="border-0 shadow-sm bg-white/70 backdrop-blur-md">
                  <CardContent className="p-4 flex flex-col justify-center items-center text-center space-y-2">
                    <Badge variant="outline" className="text-green-500 border-green-200 bg-green-50">
                      Unripe
                    </Badge>
                    <span className="text-4xl font-extrabold text-slate-700">{stats.unripe}</span>
                  </CardContent>
                </Card>
                <Card className="border-0 shadow-sm bg-white/70 backdrop-blur-md">
                  <CardContent className="p-4 flex flex-col justify-center items-center text-center space-y-2">
                    <Badge variant="secondary" className="bg-slate-100 text-slate-600 hover:bg-slate-200">
                      Total
                    </Badge>
                    <span className="text-4xl font-extrabold text-slate-900">{stats.total}</span>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>

          {/* Right Column: Video & Overlay */}
          <div className="lg:col-span-8">
            <Card className="border-0 shadow-2xl shadow-slate-200/50 bg-black overflow-hidden min-h-[300px] lg:min-h-[500px] flex items-center justify-center relative ring-1 ring-slate-200/50 rounded-xl">
              {!isStreaming ? (
                <div className="text-center p-6 lg:p-8 text-slate-400 w-full flex flex-col items-center justify-center">
                  <Camera className="w-16 h-16 lg:w-24 lg:h-24 mb-4 lg:mb-6 opacity-30" />
                  <p className="text-base lg:text-lg">Camera is off</p>
                  <p className="text-xs lg:text-sm opacity-60">
                    Click Start Webcam to begin real-time detection.
                  </p>
                </div>
              ) : (
                <div className="relative w-full h-full flex items-center justify-center bg-black p-2">
                  <div className="relative inline-block overflow-hidden">
                    {/* Hidden canvas used for reading pixels */}
                    <canvas ref={canvasRef} className="hidden" />
                    
                    <video
                      ref={videoRef}
                      className="block max-w-full max-h-[70vh] w-auto h-auto object-contain rounded-md"
                      playsInline
                      muted
                      onLoadedMetadata={updateScale}
                    />

                    {/* Bounding Boxes Overlay */}
                    {displayScale.x > 0 && predictions.map((pred, idx) => (
                      <div
                        key={idx}
                        className="absolute border-[3px] rounded-sm group transition-all"
                        style={{
                          left: `${pred.box.x1 * displayScale.x}px`,
                          top: `${pred.box.y1 * displayScale.y}px`,
                          width: `${(pred.box.x2 - pred.box.x1) * displayScale.x}px`,
                          height: `${(pred.box.y2 - pred.box.y1) * displayScale.y}px`,
                          borderColor: getClassColor(pred.class_name),
                          backgroundColor: getClassColor(pred.class_name).replace("1)", "0.15)"),
                        }}
                      >
                        <div
                          className="absolute -top-7 left-[-3px] px-2 py-1 text-[10px] font-bold text-white whitespace-nowrap rounded-t-sm rounded-br-sm shadow-sm"
                          style={{ backgroundColor: getClassColor(pred.class_name) }}
                        >
                          {pred.class_name} • {(pred.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
