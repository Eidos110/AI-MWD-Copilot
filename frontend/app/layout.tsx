import type { Metadata } from 'next';
import './globals.css';
import { Toaster } from '@/components/ui/Toast';

export const metadata: Metadata = {
  title: 'AI-Powered MWD Copilot',
  description: 'Real-time ML system for drilling decision support',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50">
        <Toaster position="top-right" />
        {children}
      </body>
    </html>
  );
}
