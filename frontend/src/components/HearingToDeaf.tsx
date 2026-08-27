'use client'

import { useState, useRef } from 'react';
import { useAvatarRenderer } from '@/hooks/use-avatar-renderer';
import { Mic, Send } from 'lucide-react';

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
    <div className="flex flex-col lg:flex-row gap-4 h-full">
      {/* Left Panel: Avatar & Visual Controls */}
      <div className="w-full lg:w-7/12 flex flex-col h-full">
          <div className="glass-card stagger-1 h-full flex flex-col overflow-hidden relative pb-3">
              <div className="card-header-bar px-4 py-3 flex justify-between items-center shrink-0">
                  <div className="flex items-center gap-2">
                      <i className="fa-solid fa-cube text-magenta"></i>
                      <span className="panel-title">3D Avatar Signer</span>
                  </div>
                  <span className="badge bg-magenta-glow py-1 px-3 rounded-full font-mono text-xs">LIVE RENDER</span>
              </div>

              {/* Avatar Box Area */}
              <div className="webcam-container flex-1 bg-black/40 m-3 rounded-2xl relative overflow-hidden border border-white/5">
                  <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
              </div>

              {/* Input Controls */}
              <div className="px-3 flex-shrink-0">
                <form onSubmit={handleTextSubmit} className="flex flex-wrap items-center gap-3 w-full">
                  <button
                    type="button"
                    onClick={startRecording}
                    disabled={isRecording}
                    className={`shrink-0 h-10 w-10 sm:w-auto sm:px-4 rounded-xl flex items-center justify-center transition-all duration-300 ${isRecording ? "bg-red-500 shadow-[0_0_20px_rgba(239,68,68,0.6)] text-white" : "btn-action-glass hover:text-cyan-400"}`}
                    title="Voice Input"
                  >
                    <Mic size={18} className={isRecording ? "animate-pulse" : "sm:mr-2"} />
                    <span className="hidden sm:inline font-semibold">{isRecording ? "Listening" : "Use Voice"}</span>
                  </button>
                  <select
                    value={speechLang}
                    onChange={(e) => setSpeechLang(e.target.value)}
                    disabled={isRecording}
                    className="select-glass bg-black/40 border border-white/10 rounded-xl px-3 h-10 text-sm font-medium text-gray-300 focus:outline-none hidden sm:block"
                  >
                    <option value="en-US">EN (English)</option>
                    <option value="hi-IN">HI (Hindi)</option>
                    <option value="mr-IN">MR (Marathi)</option>
                    <option value="bn-IN">BN (Bengali)</option>
                    <option value="ta-IN">TA (Tamil)</option>
                    <option value="gu-IN">GU (Gujarati)</option>
                  </select>
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputText}
                      onChange={(e: any) => setInputText(e.target.value)}
                      placeholder={isRecording ? "Listening..." : "Type text to translate..."}
                      className="w-full bg-white/5 border border-white/10 rounded-xl pl-4 pr-12 h-10 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 text-sm"
                      disabled={isRecording}
                    />
                    <button
                      type="submit"
                      disabled={!inputText.trim() || loading}
                      className="absolute right-1 top-1 bottom-1 w-8 rounded-lg btn-magenta-glow flex items-center justify-center disabled:opacity-50"
                    >
                      <Send size={14} />
                    </button>
                  </div>
                </form>
              </div>
          </div>
      </div>

      {/* Right Panel: Output, Translation, Gloss */}
      <div className="w-full lg:w-5/12 flex flex-col gap-4 h-full">
          
          {/* Transcript Panel */}
          <div className="glass-card stagger-2 p-5 relative overflow-hidden shrink-0 min-h-[140px] flex flex-col justify-center">
              <div className="glow-accent-left"></div>
              <span className="small-label-accent">ORIGINAL TRANSCRIPT</span>
              <div className="mt-3 flex items-center">
                  <h2 className="text-xl font-medium text-white break-words w-full leading-tight">
                      {transcript || <span className="text-gray-500 italic text-lg">Waiting for voice or text input...</span>}
                  </h2>
              </div>
          </div>

          {/* ISL Gloss Panel */}
          <div className="glass-card stagger-3 p-5 shrink-0 relative min-h-[160px] flex flex-col">
              <div className="flex justify-between items-center mb-3">
                  <span className="panel-sub-title flex items-center"><i className="fa-solid fa-language text-cyan mr-2"></i>ISL Gloss Translation</span>
                  {loading && <span className="badge bg-cyan-glow animate-pulse py-1 px-2 rounded-md text-[10px]">TRANSLATING</span>}
              </div>
              <div className="flex-1 bg-black/20 rounded-xl p-4 border border-white/5 flex items-center shadow-inner overflow-hidden">
                <span className="text-2xl font-black text-cyan tracking-wider uppercase leading-tight break-words">
                  {islGloss || <span className="text-gray-600 font-medium">---</span>}
                </span>
              </div>
          </div>

          {/* Current Sign Panel */}
          <div className="glass-card stagger-4 p-5 flex-1 flex flex-col min-h-[180px]">
              <div className="flex justify-between items-center mb-3 shrink-0">
                  <span className="panel-sub-title flex items-center"><i className="fa-solid fa-play text-magenta mr-2"></i>Active Animation</span>
              </div>
              <div className="flex-1 flex flex-col items-center justify-center bg-emerald-500/5 rounded-xl border border-emerald-500/20 relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent opacity-50" />
                  {currentSign ? (
                      <span className="text-4xl font-black text-emerald-400 tracking-widest uppercase animate-pulse drop-shadow-[0_0_15px_rgba(16,185,129,0.3)] text-center px-4">
                          {currentSign}
                      </span>
                  ) : (
                      <div className="flex flex-col items-center text-gray-600 opacity-60">
                        <i className="fa-solid fa-person-rays text-3xl mb-2"></i>
                        <span className="font-semibold tracking-[0.2em] uppercase text-xs">IDLE STATE</span>
                      </div>
                  )}
              </div>
          </div>
      </div>
    </div>
  );
}
