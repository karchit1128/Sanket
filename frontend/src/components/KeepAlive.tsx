'use client'
import { useEffect } from 'react';

export default function KeepAlive() {
  useEffect(() => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    
    // Ping backend to wake it up and prevent Render/Vercel from sleeping
    const ping = () => {
      fetch(backendUrl).catch(() => {
        // Ignore failures, this is just a wake-up ping
      });
    };
    
    // Ping immediately on mount
    ping();
    
    // Ping every 14 minutes (Render sleeps after 15 mins of inactivity)
    const interval = setInterval(ping, 14 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  return null;
}
