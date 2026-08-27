import type { Metadata } from 'next';
import { Outfit } from 'next/font/google';
import './globals.css';

const outfit = Outfit({ subsets: ['latin'], variable: '--font-outfit' });

export const metadata: Metadata = {
  title: 'Project Sanket | ISL Translator',
  description: 'Bi-directional Indian Sign Language Translator for SIH',
};

import KeepAlive from '@/components/KeepAlive';

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${outfit.variable} dark`}>
      <body className="font-outfit antialiased bg-[#05050a] text-white min-h-screen">
        <KeepAlive />
        {children}
      </body>
    </html>
  );
}
