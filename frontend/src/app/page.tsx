'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

export default function LandingPage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <>
      <div className="mesh-bg"></div>
      <div className="gradient-overlay"></div>

      <div className="container mx-auto py-8 px-4 sm:px-6 lg:px-8 max-w-[1200px] min-h-screen flex flex-col justify-center relative z-10">
        
        {/* Hero Section */}
        <div className="text-center mb-12 stagger-1 flex flex-col items-center">
          <div className="inline-flex items-center justify-center hero-logo-orb shadow-glow mx-auto overflow-hidden bg-transparent">
            <img src="/logo.png" alt="Sanket Logo" className="w-full h-full object-cover" />
          </div>
          <h1 className="hero-logo-text mb-3 tracking-tight drop-shadow-lg">
            Sanket
          </h1>
          <p className="text-lg md:text-2xl text-gray-300 max-w-2xl mx-auto font-light leading-relaxed">
            Bridging the communication gap with <span className="text-cyan font-semibold">Real-Time AI</span>
          </p>
        </div>

        {/* Feature Navigation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto w-full stagger-2">
          
          {/* Card 1: Hearing to Deaf */}
          <Link href="/hearing-to-deaf" className="group block">
            <div className="glass-card h-full p-6 rounded-3xl transition-all duration-500 hover:-translate-y-2 hover:shadow-[0_0_25px_rgba(255,0,255,0.25)] border border-white/5 hover:border-magenta/50 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-28 h-28 bg-magenta/20 rounded-full blur-3xl -mr-12 -mt-12 transition-all group-hover:bg-magenta/40"></div>
              
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-magenta to-purple-600 flex items-center justify-center text-xl text-white mb-4 shadow-lg">
                <i className="fa-solid fa-ear-listen"></i>
              </div>
              
              <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                Hearing <i className="fa-solid fa-arrow-right-long text-magenta text-lg"></i> Deaf
              </h2>
              
              <p className="text-gray-400 text-base mb-5 line-clamp-3">
                Convert spoken English or Hindi voice and text into real-time, highly accurate 3D Sign Language animations.
              </p>

              <div className="flex items-center text-magenta font-semibold text-sm group-hover:text-white transition-colors">
                Launch Avatar Engine <i className="fa-solid fa-arrow-right ml-2 group-hover:translate-x-1 transition-transform"></i>
              </div>
            </div>
          </Link>

          {/* Card 2: Deaf to Hearing */}
          <Link href="/deaf-to-hearing" className="group block">
            <div className="glass-card h-full p-6 rounded-3xl transition-all duration-500 hover:-translate-y-2 hover:shadow-[0_0_25px_rgba(0,255,255,0.25)] border border-white/5 hover:border-cyan/50 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-28 h-28 bg-cyan/20 rounded-full blur-3xl -mr-12 -mt-12 transition-all group-hover:bg-cyan/40"></div>
              
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan to-blue-600 flex items-center justify-center text-xl text-white mb-4 shadow-lg">
                <i className="fa-solid fa-hand-sparkles"></i>
              </div>
              
              <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                Deaf <i className="fa-solid fa-arrow-right-long text-cyan text-lg"></i> Hearing
              </h2>
              
              <p className="text-gray-400 text-base mb-5 line-clamp-3">
                Use your webcam to detect sign language gestures and instantly translate them into spoken audio and text.
              </p>

              <div className="flex items-center text-cyan font-semibold text-sm group-hover:text-white transition-colors">
                Launch Vision Engine <i className="fa-solid fa-arrow-right ml-2 group-hover:translate-x-1 transition-transform"></i>
              </div>
            </div>
          </Link>

        </div>

        {/* Feature Highlights */}
        <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-5 max-w-4xl mx-auto text-center stagger-3">
          <div className="glass-card p-5 rounded-2xl border-white/5">
            <i className="fa-solid fa-bolt text-xl text-yellow-400 mb-2"></i>
            <h3 className="text-white font-semibold mb-1 text-sm">Real-Time</h3>
            <p className="text-gray-400 text-xs">Under 50ms latency for seamless conversations.</p>
          </div>
          <div className="glass-card p-5 rounded-2xl border-white/5">
            <i className="fa-solid fa-language text-xl text-green-400 mb-2"></i>
            <h3 className="text-white font-semibold mb-1 text-sm">Bilingual</h3>
            <p className="text-gray-400 text-xs">Native support for English and Hindi input processing.</p>
          </div>
          <div className="glass-card p-5 rounded-2xl border-white/5">
            <i className="fa-solid fa-cube text-xl text-purple-400 mb-2"></i>
            <h3 className="text-white font-semibold mb-1 text-sm">3D Avatars</h3>
            <p className="text-gray-400 text-xs">State-of-the-art rigging and blendshapes for fluid signs.</p>
          </div>
        </div>

      </div>
    </>
  );
}
