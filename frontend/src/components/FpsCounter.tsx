'use client'

import { useState, useEffect } from 'react';

export default function FpsCounter() {
  const [fps, setFps] = useState(60);

  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationFrameId: number;

    const calculateFps = (currentTime: number) => {
      frameCount++;
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }
      animationFrameId = requestAnimationFrame(calculateFps);
    };

    animationFrameId = requestAnimationFrame(calculateFps);

    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  return (
    <div className="status-badge hidden sm:flex">
      <span className="badge-label text-cyan font-mono mr-2">{fps.toFixed(1)}</span> FPS
    </div>
  );
}
