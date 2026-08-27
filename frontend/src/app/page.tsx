'use client'

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
            <div className="flex items-center">
                <div className="logo-orb mr-4 text-xl">
                    <i className="fa-solid fa-ear-listen" />
                </div>
                <div>
                    <h1 className="logo-text mb-0 text-2xl tracking-tighter">Sanket</h1>
                    <p className="subtitle-text mb-0">Hearing to Deaf &bull; Voice/Text to 3D Sign Language AI</p>
                </div>
            </div>
            
            <div className="flex items-center gap-4 mt-3 md:mt-0">
                <div className="status-badge" id="cameraStatusBadge">
                    <span className="pulse-dot bg-success"></span>
                    <span className="badge-label">Engine Active</span>
                </div>
                
                {/* Live FPS Counter */}
                <FpsCounter />

                <a 
                  href={process.env.NEXT_PUBLIC_AURA_URL || "http://127.0.0.1:5000"} 
                  className="btn-action-glass hover:text-cyan-400 px-4 py-2 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all duration-300 ml-2"
                  title="Switch to Deaf -> Hearing Mode"
                >
                  Switch to Deaf ➔ Hearing
                </a>
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
