'use client'

import { useState } from 'react';
import HearingToDeaf from '@/components/HearingToDeaf';
import DeafToHearing from '@/components/DeafToHearing';

export default function Home() {
  const [activeMode, setActiveMode] = useState<'avatar' | 'camera'>('avatar');

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
              Two-Way Neural ISL Bridge
            </p>
          </div>
          
          <div className="flex gap-3 hidden md:flex">
            <div className="flex items-center gap-2 bg-black/40 px-4 py-2 rounded-full border border-white/5 backdrop-blur-md">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-bold text-gray-300 uppercase tracking-wider">Language Core Active</span>
            </div>
            <div className="flex items-center gap-2 bg-black/40 px-4 py-2 rounded-full border border-white/5 backdrop-blur-md">
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              <span className="text-xs font-bold text-gray-300 uppercase tracking-wider">Vision Core Active</span>
            </div>
          </div>
        </div>

        {/* Custom Toggle Switch */}
        <div className="w-full max-w-md bg-black/40 border border-white/10 rounded-full p-1 mb-8 flex relative z-10 backdrop-blur-md">
          <button
            onClick={() => setActiveMode('avatar')}
            className={`flex-1 py-3 px-6 rounded-full text-sm font-bold tracking-widest uppercase transition-all duration-300 z-10 ${
              activeMode === 'avatar' ? 'text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            Avatar Mode
          </button>
          <button
            onClick={() => setActiveMode('camera')}
            className={`flex-1 py-3 px-6 rounded-full text-sm font-bold tracking-widest uppercase transition-all duration-300 z-10 ${
              activeMode === 'camera' ? 'text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            Camera Mode
          </button>
          
          {/* Active indicator pill */}
          <div 
            className={`absolute top-1 bottom-1 w-[calc(50%-4px)] rounded-full transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] ${
              activeMode === 'avatar' 
                ? 'left-1 bg-gradient-to-r from-emerald-600 to-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.3)]' 
                : 'left-[calc(50%+3px)] bg-gradient-to-r from-blue-600 to-blue-400 shadow-[0_0_15px_rgba(37,99,235,0.3)]'
            }`} 
          />
        </div>

        {/* Dynamic Main Layout */}
        <div className="w-full max-w-5xl h-[65vh] relative">
          <div className={`absolute inset-0 transition-opacity duration-500 ${
            activeMode === 'avatar' ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
          }`}>
            <HearingToDeaf />
          </div>
          <div className={`absolute inset-0 transition-opacity duration-500 ${
            activeMode === 'camera' ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
          }`}>
            <DeafToHearing isVisible={activeMode === 'camera'} />
          </div>
        </div>
        
      </main>
    </>
  );
}
