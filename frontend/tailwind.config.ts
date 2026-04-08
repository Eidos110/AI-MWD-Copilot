import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#1f77b4',
          600: '#1a6a9e',
          700: '#155d88',
          800: '#105072',
          900: '#0b435c',
        },
        well: {
          gr: '#22c55e',
          resistivity: '#ef4444',
          porosity: '#3b82f6',
          predicted: '#f97316',
          pressure: '#a855f7',
          fluid: {
            background: '#9ca3af',
            payzone: '#eab308',
            reservoir: '#f87171',
          },
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};

export default config;
