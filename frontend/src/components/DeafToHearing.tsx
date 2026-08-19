'use client'

import { useState, useRef, useEffect } from 'react';
import { Camera, Scan } from 'lucide-react';

export default function DeafToHearing({ isVisible }: { isVisible: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [deafTranslation, setDeafTranslation] = useState('');
  const [status, setStatus] = useState('idle');
  const [indicator, setIndicator] = useState('red');

  useEffect(() => {
    if (deafTranslation && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(deafTranslation);
      window.speechSynthesis.speak(utterance);
      const timeout = setTimeout(() => { setDeafTranslation(''); }, 3000);
      return () => clearTimeout(timeout);
    }
  }, [deafTranslation]);

  useEffect(() => {
    if (!isVisible && isCameraActive) {
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      if (wsRef.current) wsRef.current.close();
      setIsCameraActive(false);
      setDeafTranslation('');
    }
  }, [isVisible, isCameraActive]);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, []);

  const startCamera = async () => {
    if (isCameraActive) {
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      if (wsRef.current) wsRef.current.close();
      setIsCameraActive(false);
      setDeafTranslation('');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current!.play();
          setIsCameraActive(true);
          const wsUrl = `ws://${window.location.hostname}:8000/ws/video`;
          wsRef.current = new WebSocket(wsUrl);
          wsRef.current.onopen = () => { sendFrames(); };
          wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status) setStatus(data.status);
            if (data.indicator) setIndicator(data.indicator);
            if (data.prediction) setDeafTranslation(data.prediction);
            
            // Handle flash effect when prediction fires
            if (data.prediction_fired) {
               setIndicator('flash');
               setTimeout(() => setIndicator('green'), 300);
            }
            
            const ctx = overlayRef.current?.getContext('2d');
            if (ctx && overlayRef.current) {
              ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
              if (data.landmarks && data.landmarks.length > 0) {
                ctx.fillStyle = '#3b82f6';
                data.landmarks.forEach((lm: any) => {
                  ctx.beginPath();
                  ctx.arc(lm.x * 640, lm.y * 480, 4, 0, 2 * Math.PI);
                  ctx.fill();
                });
              }
            }
          };
        };
      }
    } catch (err) {
      alert('Could not access camera. Please allow permissions.');
    }
  };

  const sendFrames = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, 640, 480);
      canvasRef.current.toBlob((blob) => {
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          const reader = new FileReader();
          reader.onloadend = () => {
            wsRef.current?.send(reader.result as string);
          };
          reader.readAsDataURL(blob);
        }
      }, 'image/jpeg', 0.8);
    }
    setTimeout(() => requestAnimationFrame(sendFrames), 100);
  };

  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="flex-1 glass-panel rounded-3xl overflow-hidden relative flex flex-col justify-center items-center">
        <div className="absolute top-4 left-4 z-10 glass-panel px-4 py-2 rounded-full text-xs font-bold text-blue-400 uppercase tracking-widest flex items-center gap-2">
          <Camera size={14} /> Deaf to Hearing
        </div>

        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ opacity: isCameraActive ? 1 : 0, zIndex: isCameraActive ? 0 : -10 }}
          className="absolute inset-0 w-full h-full object-cover -scale-x-100"
        />

        {isCameraActive ? (
          <>
            <canvas ref={overlayRef} width={640} height={480} className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10" />
            
            {/* Status Indicator Light */}
            <div className="absolute top-6 right-6 z-30 flex items-center gap-3 glass-panel px-4 py-2 rounded-full shadow-lg">
              <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">{status}</span>
              <div className={`w-4 h-4 rounded-full shadow-[0_0_10px_currentColor] transition-all duration-300 ${
                indicator === 'red' ? 'bg-red-500 text-red-500' : 
                indicator === 'yellow' ? 'bg-yellow-400 text-yellow-400' :
                indicator === 'green' ? 'bg-green-500 text-green-500' :
                indicator === 'pulsing' ? 'bg-blue-500 text-blue-500 animate-pulse scale-125' :
                indicator === 'flash' ? 'bg-white text-white scale-150 shadow-[0_0_30px_#fff]' :
                'bg-gray-500 text-gray-500'
              }`} />
            </div>

            {/* Hand Zone Overlay */}
            <div className={`absolute top-[20%] left-[10%] right-[10%] bottom-[20%] pointer-events-none z-20 border-2 rounded-3xl transition-all duration-300 ${
              indicator === 'pulsing' ? 'border-blue-500/80 bg-blue-500/5 shadow-[0_0_30px_rgba(59,130,246,0.3)_inset]' : 
              indicator === 'flash' ? 'border-white bg-white/20' : 
              'border-blue-500/20 border-dashed'
            }`} />
            
          </>
        ) : (
          <div className="text-center flex flex-col items-center gap-4 text-blue-400/60 p-8">
            <div className="w-20 h-20 rounded-full border-2 border-dashed border-blue-400/30 flex items-center justify-center">
              <Scan size={32} />
            </div>
            <p className="text-sm font-semibold uppercase tracking-widest">Camera Offline</p>
          </div>
        )}
        <canvas ref={canvasRef} width="640" height="480" className="hidden" />
      </div>

      <div className="h-32 glass-panel rounded-3xl p6 flex items-center justify-between gap-6">
        <button
          onClick={startCamera}
          className={'shrink-0 w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 ' + (isCameraActive ? 'bg-red-500 hover:bg-red-600 shadow-[0_0_20px_rgba(239,68,68,0.4)]' : 'bg-blue-600 hover:bg-blue-500 hover:scale-105 shadow-[0_0_20px_rgba(37,99,235,0.3)]')}
        >
          {isCameraActive ? <Scan size={28} className="text-white" /> : <Camera size={28} className="text-white" />}
        </button>
        <div className="flex-1 flex flex-col justify-center min-w-0">
          <span className="text-xs text-gray-400 uppercase tracking-widest mb-1 font-semibold">AI Vision Detection</span>
          <p className="text-lg text-white truncate font-bold tracking-wider">
            {deafTranslation ? deafTranslation.toUpperCase() : (isCameraActive ? (status === 'detecting' ? 'DETECTING...' : 'AI ACTIVE') : 'Activate AI Vision')}
          </p>
        </div>
      </div>
    </div>
  );
}
