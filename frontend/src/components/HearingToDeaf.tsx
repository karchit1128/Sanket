'use client'

import { useState, useRef, useEffect } from 'react';
import { useAvatarRenderer } from '@/hooks/use-avatar-renderer';
import { Mic, VolumeX, Send } from 'lucide-react';

export default function HearingToDeaf() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [islGloss, setIslGloss] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentSign, setCurrentSign] = useState("");
  const [inputText, setInputText] = useState("");
  const [speechLang, setSpeechLang] = useState("en-US");

  const containerRef = useRef<HTMLDivElement>(null);

  const { executeSignSequence } = useAvatarRenderer({
    containerRef: containerRef,
    modelPath: "/ybot.glb",
    animationSpeed: 1.2,
    pauseDuration: 300,
    onTextUpdate: (text) => setCurrentSign((prev) => prev + text),
  });

  const startRecording = () => {
    // @ts-ignore
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Your browser does not support Speech Recognition. Try Chrome.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = speechLang;
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

  const handleTextSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const textToTranslate = inputText.trim();
    if (!textToTranslate) return;

    setInputText(""); // Clear input instantly
    setTranscript(textToTranslate);
    await translateToISL(textToTranslate);
  };

  const translateToISL = async (text: string) => {
    setLoading(true);
    setCurrentSign(""); // Reset the live sign text
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setIslGloss(data.gloss);
      // Force avatar to execute sign even if gloss is identical to previous
      if (containerRef.current && data.gloss) {
        executeSignSequence(data.gloss);
      }
    } catch (error) {
      console.error("Translation failed", error);
      setIslGloss("ERROR");
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="flex-1 glass-panel rounded-3xl overflow-hidden relative group">
        <div className="absolute top-4 left-4 z-10 glass-panel px-4 py-2 rounded-full text-xs font-bold text-emerald-400 uppercase tracking-widest flex items-center gap-2">
          <VolumeX size={14} /> Hearing to Deaf
        </div>
        <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

        {/* Transcript & Gloss Output Overlay */}
        {(transcript || islGloss) && (
          <div className="absolute bottom-6 w-full px-6 flex justify-center z-10 transition-all duration-500">
            <div className="glass-panel bg-black/60 px-8 py-4 rounded-2xl flex flex-col items-center">
              {transcript && (
                <>
                  <span className="text-[10px] text-blue-400 uppercase tracking-widest mb-1">Original Text</span>
                  <span className="text-lg text-gray-300 mb-3 text-center max-w-lg font-medium">{transcript}</span>
                </>
              )}
              {islGloss && (
                <>
                  <span className="text-[10px] text-emerald-400 uppercase tracking-widest mb-1">ISL Gloss</span>
                  <span className="text-2xl font-bold text-white tracking-widest text-center">{islGloss}</span>
                </>
              )}
              {currentSign && (
                <div className="mt-4 px-8 py-3 bg-emerald-500/20 border border-emerald-500/50 rounded-xl shadow-[0_0_20px_rgba(16,185,129,0.2)]">
                  <span className="text-[10px] text-emerald-300 uppercase tracking-widest mb-1 block text-center">Currently Signing</span>
                  <span className="text-4xl font-black text-emerald-400 tracking-[0.2em]">{currentSign}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Input Controls */}
      <div className="glass-panel rounded-3xl p-6 flex items-center justify-between gap-4 relative overflow-hidden">
        {loading && <div className="absolute inset-0 bg-blue-500/10 animate-pulse" />}

        <div className="flex items-center gap-3 z-10">
          <button
            onClick={startRecording}
            disabled={isRecording}
            className={`shrink-0 w-14 h-14 rounded-full flex items-center justify-center transition-all duration-300 ${isRecording
                ? "bg-red-500 shadow-[0_0_30px_rgba(239,68,68,0.6)] scale-110"
                : "bg-emerald-600 hover:bg-emerald-500 hover:scale-105 shadow-[0_0_20px_rgba(16,185,129,0.3)]"
              }`}
            title="Voice Input"
          >
            <Mic size={24} className="text-white" />
          </button>

          <select
            value={speechLang}
            onChange={(e) => setSpeechLang(e.target.value)}
            disabled={isRecording}
            className="bg-black/40 border border-white/10 rounded-xl px-3 py-2 text-sm font-semibold text-gray-300 focus:outline-none focus:border-emerald-500/50 cursor-pointer hover:bg-black/60 transition-colors"
            title="Select Voice Language"
          >
            <option value="en-US">EN (English)</option>
            <option value="hi-IN">HI (Hindi)</option>
            <option value="mr-IN">MR (Marathi)</option>
            <option value="bn-IN">BN (Bengali)</option>
            <option value="ta-IN">TA (Tamil)</option>
            <option value="gu-IN">GU (Gujarati)</option>
          </select>
        </div>

        <form onSubmit={handleTextSubmit} className="flex-1 flex items-center gap-4 z-10">
          <input
            type="text"
            value={inputText}
            onChange={(e: any) => setInputText(e.target.value)}
            placeholder={isRecording ? "Listening..." : "Type here or use voice..."}
            className="flex-1 bg-white/5 border border-white/10 rounded-2xl px-6 py-4 text-white placeholder:text-gray-500 focus:outline-none focus:border-emerald-500/50 transition-colors"
            disabled={isRecording}
          />
          <button
            type="submit"
            disabled={!inputText.trim() || loading}
            className="shrink-0 w-14 h-14 rounded-2xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 flex items-center justify-center transition-colors shadow-[0_0_20px_rgba(37,99,235,0.3)]"
          >
            <Send size={24} className="text-white" />
          </button>
        </form>
      </div>
    </div>
  );
}
