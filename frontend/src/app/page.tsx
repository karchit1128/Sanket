'use client'

import HearingToDeaf from '@/components/HearingToDeaf';

export default function Home() {
  return (
    <>
      <div className="mesh-bg" />
      <main className="flex min-h-screen flex-col items-center justify-center p-4 md:p-8">
        
        {/* Header */}
        <div className="w-full max-w-5xl flex items-center justify-between mb-8 px-4">
          <div>
            <h1 className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-500 mb-2 tracking-tighter drop-shadow-lg">
              SANKET.AI
            </h1>
            <p className="text-sm font-semibold tracking-[0.2em] text-gray-400 uppercase">
              ISL 3D Avatar Translator
            </p>
          </div>
          
          <div className="flex gap-3 hidden md:flex">
            <div className="flex items-center gap-2 bg-black/40 px-4 py-2 rounded-full border border-white/5 backdrop-blur-md">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-bold text-gray-300 uppercase tracking-wider">Avatar Engine Active</span>
            </div>
            <a 
              href={process.env.NEXT_PUBLIC_AURA_URL || "http://localhost:5000"} 
              target="_blank" 
              rel="noreferrer"
              className="flex items-center gap-2 bg-blue-600/30 hover:bg-blue-600/50 px-4 py-2 rounded-full border border-blue-400/30 backdrop-blur-md transition-all text-xs font-bold text-blue-300 uppercase tracking-wider"
            >
              Open Camera Mode (AuraGesture) ↗
            </a>
          </div>
        </div>

        {/* Avatar Component View */}
        <div className="w-full max-w-5xl h-[70vh] relative">
          <HearingToDeaf />
        </div>
        
      </main>
    </>
  );
}

