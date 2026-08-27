'use client'

import Link from 'next/link';
import HearingToDeaf from '@/components/HearingToDeaf';
import FpsCounter from '@/components/FpsCounter';

export default function Home() {
  return (
    <>
      <div className="mesh-bg"></div>
      <div className="gradient-overlay"></div>

      <div className="container mx-auto py-4 px-4 sm:px-6 lg:px-8 max-w-[1400px] h-screen flex flex-col relative z-10">
        {/* Floating Navigation Header */}
        <header className="glass-card mb-4 px-6 py-4 flex flex-wrap justify-between items-center shrink-0">
            <Link href="/" className="flex items-center hover:opacity-80 transition-opacity">
                <div className="mr-4 flex-shrink-0">
                    <img src="/logo.png" alt="Sanket Logo" className="w-12 h-12 rounded-xl object-cover shadow-[0_0_15px_rgba(34,211,238,0.3)]" />
                </div>
                <div>
                    <h1 className="logo-text mb-0 text-2xl tracking-tighter">Sanket</h1>
                    <p className="subtitle-text mb-0">Hearing to Deaf &bull; Voice/Text to 3D Sign Language AI</p>
                </div>
            </Link>
            
            <div className="flex items-center gap-4 mt-3 md:mt-0">
                <div className="status-badge" id="cameraStatusBadge">
                    <span className="pulse-dot bg-success"></span>
                    <span className="badge-label">Engine Active</span>
                </div>
                
                {/* Live FPS Counter */}
                <FpsCounter />

                <Link 
                  href="/deaf-to-hearing" 
                  className="btn-action-glass hover:text-cyan-400 px-4 py-2 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all duration-300 ml-2"
                  title="Switch to Deaf -> Hearing Mode"
                >
                  Switch to Deaf ➔ Hearing
                </Link>
            </div>
        </header>

        {/* Main Dashboard Workspace */}
        <div className="flex-1 min-h-0 pb-4">
          <HearingToDeaf />
        </div>
      </div>
    </>
  );
}
