'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

export default function DeafToHearing() {
  const [iframeUrl, setIframeUrl] = useState('');
  const router = useRouter();

  useEffect(() => {
    // In deployment, set NEXT_PUBLIC_AURA_URL in Vercel to your deployed Django URL
    const url = process.env.NEXT_PUBLIC_AURA_URL || 'http://127.0.0.1:5000';
    setIframeUrl(url);

    // Listen for navigation messages from the iframe
    const handleMessage = (event: MessageEvent) => {
      if (event.data === 'SWITCH_TO_HEARING_TO_DEAF') {
        router.push('/hearing-to-deaf');
      } else if (event.data === 'GO_HOME') {
        router.push('/');
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [router]);

  return (
    <div className="w-full h-screen bg-black overflow-hidden m-0 p-0">
      {iframeUrl && (
        <iframe 
          src={iframeUrl} 
          className="w-full h-full border-none m-0 p-0"
          allow="camera; microphone"
          title="Deaf to Hearing AI"
        ></iframe>
      )}
    </div>
  );
}
