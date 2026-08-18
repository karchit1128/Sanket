import type { Metadata } from 'next';
import { Outfit } from 'next/font/google';
import './globals.css';

const outfit = Outfit({ subsets: ['latin'], variable: '--font-outfit' });

export const metadata: Metadata = {
  title: 'Project Sanket | ISL Translator',
  description: 'Bi-directional Indian Sign Language Translator for SIH',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="${outfit.variable} dark">
      <body className="font-outfit antialiased bg-[#05050a] text-white min-h-screen">
        {children}
      </body>
    </html>
  );
}
