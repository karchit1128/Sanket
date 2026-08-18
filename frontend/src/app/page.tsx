'use client'

import { useState, useRef, useEffect } from 'react';
import { useAvatarRenderer } from '@/hooks/use-avatar-renderer';
import { Mic, Video, VolumeX, MessageSquare } from 'lucide-react';

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [islGloss, setIslGloss] = useState("");
  const [loading, setLoading] = useState(false);
  
  // Webcam & WebSocket states
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [deafTranslation, setDeafTranslation] = useState("");

  // The avatar container
  const containerRef = useRef<HTMLDivElement>(null);

  const { executeSignSequence } = useAvatarRenderer({
    containerElement: containerRef.current,
    modelPath: "/ybot.glb",
    animationSpeed: 1.2,
    pauseDuration: 300,
  });

  useEffect(() => {
    if (islGloss && containerRef.current) {
      executeSignSequence(islGloss);
    }
  }, [islGloss, executeSignSequence]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsCameraActive(true);
      }
      wsRef.current = new WebSocket("ws://localhost:8000/ws/video");
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.prediction) {
          setDeafTranslation(data.prediction);
        }
      };
      sendFrames();
    } catch (err) {
      console.error("Camera access denied", err);
      alert("Please allow camera access.");
    }
  };

  const sendFrames = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState === WebSocket.OPEN) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
        const frameData = canvasRef.current.toDataURL('image/jpeg', 0.5);
        wsRef.current.send(frameData);
      }
    }
    setTimeout(() => {
      requestAnimationFrame(sendFrames);
    }, 100);
  };

  const startRecording = () => {
    // @ts-ignore
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Your browser does not support Speech Recognition. Try Chrome.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.onstart = () => setIsRecording(true);
    recognition.onresult = async (event: any) => {
      const text = event.results[0][0].transcript;
      setTranscript(text);
      await translateToISL(text);
    };
    recognition.onerror = () => setIsRecording(false);
    recognition.onend = () => setIsRecording(false);
    recognition.start();
  };

  const translateToISL = async (text: string) => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setIslGloss(data.gloss);
    } catch (error) {
      console.error("Translation failed", error);
      setIslGloss("ERROR");
    }
    setLoading(false);
  };

  return (
    <>
      <div className="mesh-bg" />
      <main className="flex min-h-screen flex-col items-center justify-center p-4 md:p-8">
        
        {/* Header */}
        <div className="w-full max-w-7xl flex items-center justify-between mb-8 px-4">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 text-transparent bg-clip-text glow-text tracking-tight">
              Project Sanket
            </h1>
            <p className="text-gray-400 text-sm md:text-base mt-2 tracking-wide uppercase">Bi-Directional ISL Communication System</p>
          </div>
          <div className="hidden md:flex gap-4">
            <div className="px-4 py-2 glass-panel rounded-full text-xs text-blue-300 font-semibold flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" /> Groq LLM Active
            </div>
            <div className="px-4 py-2 glass-panel rounded-full text-xs text-emerald-300 font-semibold flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" /> 3D Engine Live
            </div>
          </div>
        </div>

        {/* Main Grid Layout */}
        <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 gap-8 h-[75vh]">
          
          {/* Left Column: Hearing to Deaf (Avatar) */}
          <div className="flex flex-col gap-6 h-full">
            <div className="flex-1 glass-panel rounded-3xl overflow-hidden relative group">
              <div className="absolute top-4 left-4 z-10 glass-panel px-4 py-2 rounded-full text-xs font-bold text-emerald-400 uppercase tracking-widest flex items-center gap-2">
                <VolumeX size={14} /> Hearing to Deaf
              </div>
              <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
              
              {/* Gloss Output Overlay */}
              {islGloss && (
                <div className="absolute bottom-6 w-full px-6 flex justify-center z-10 transition-all duration-500">
                  <div className="glass-panel bg-black/60 px-8 py-4 rounded-2xl flex flex-col items-center">
                    <span className="text-[10px] text-emerald-400 uppercase tracking-widest mb-1">Translating to ISL Gloss</span>
                    <span className="text-2xl font-bold text-white tracking-widest">{islGloss}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Microphone Control */}
            <div className="h-32 glass-panel rounded-3xl p-6 flex items-center justify-between gap-6 relative overflow-hidden">
              {loading && <div className="absolute inset-0 bg-blue-500/10 animate-pulse" />}
              
              <button
                onClick={startRecording}
                disabled={isRecording}
                className={`shrink-0 w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 z-10 ${
                  isRecording 
                    ? "bg-red-500 shadow-[0_0_30px_rgba(239,68,68,0.6)] scale-110" 
                    : "bg-emerald-600 hover:bg-emerald-500 hover:scale-105 shadow-[0_0_20px_rgba(16,185,129,0.3)]"
                }`}
              >
                <Mic size={32} className="text-white" />
              </button>

              <div className="flex-1 flex flex-col justify-center min-w-0 z-10">
                <span className="text-xs text-gray-400 uppercase tracking-widest mb-1 font-semibold">Voice Input</span>
                <p className="text-lg text-white truncate opacity-90">
                  {isRecording ? "Listening..." : transcript || "Press mic to speak"}
                </p>
              </div>
            </div>
          </div>

          {/* Right Column: Deaf to Hearing (Webcam) */}
          <div className="glass-panel rounded-3xl p-2 relative flex flex-col h-full overflow-hidden group">
            <div className="absolute top-6 left-6 z-10 glass-panel px-4 py-2 rounded-full text-xs font-bold text-blue-400 uppercase tracking-widest flex items-center gap-2">
              <Video size={14} /> Deaf to Hearing
            </div>

            <div className="w-full flex-1 rounded-[1.25rem] bg-black/50 overflow-hidden relative flex items-center justify-center border border-gray-800/50">
              {!isCameraActive ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-blue-500/10 flex items-center justify-center">
                    <Video size={24} className="text-blue-400" />
                  </div>
                  <p className="text-gray-400 text-sm max-w-[250px] text-center">Enable your camera to let the AI analyze your sign language in real-time.</p>
                  <button 
                    onClick={startCamera}
                    className="mt-2 bg-blue-600 hover:bg-blue-500 px-6 py-2.5 rounded-full text-sm font-bold transition-all shadow-[0_0_20px_rgba(37,99,235,0.4)] hover:shadow-[0_0_30px_rgba(37,99,235,0.6)]"
                  >
                    Activate AI Vision
                  </button>
                </div>
              ) : (
                <>
                  <video ref={videoRef} className="w-full h-full object-cover transform scale-x-[-1]" />
                  <canvas ref={canvasRef} width="640" height="480" className="hidden" />
                  
                  {/* Camera Scanning UI Elements */}
                  <div className="absolute inset-0 border-2 border-blue-500/20 rounded-[1.25rem] m-4 pointer-events-none" />
                  
                  {/* Phase 6: Camera Calibration Silhouette Box */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none opacity-30 mt-8">
                    <div className="w-24 h-24 border-2 border-dashed border-white rounded-[3rem] mb-2" />
                    <div className="w-56 h-64 border-2 border-dashed border-white rounded-[2rem]" />
                    <span className="text-[10px] uppercase font-bold text-white tracking-widest mt-4">Align Head & Shoulders</span>
                  </div>
                  <div className="absolute inset-0 border-2 border-blue-500/20 rounded-[1.25rem] m-4 pointer-events-none" />
                  <div className="absolute top-1/2 left-0 w-full h-[1px] bg-blue-500/30 animate-[scan_3s_ease-in-out_infinite] shadow-[0_0_10px_rgba(59,130,246,0.5)] pointer-events-none" />
                  
                  {/* Real-time Translation Overlay */}
                  <div className="absolute bottom-8 w-full px-8 flex justify-center z-10">
                    <div className="glass-panel bg-black/70 px-8 py-5 rounded-2xl flex items-center gap-4 border border-blue-500/30 w-full max-w-sm">
                      <MessageSquare size={24} className="text-blue-400 shrink-0" />
                      <div className="flex flex-col min-w-0">
                        <span className="text-[10px] text-blue-400 uppercase tracking-widest mb-1">AI Prediction</span>
                        <span className="text-xl font-bold text-white uppercase tracking-widest truncate">
                          {deafTranslation || "Detecting..."}
                        </span>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

        </div>
      </main>
      
      
    </>
  );
}







