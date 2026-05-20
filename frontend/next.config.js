/** @type {import('next').NextConfig} */
const RAILWAY_BACKEND = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: RAILWAY_BACKEND,
  },
  rewrites: async () => {
    return [
      {
        source: '/api/:path*',
        destination: `${RAILWAY_BACKEND}/api/:path*`,
      },
      {
        source: '/health',
        destination: `${RAILWAY_BACKEND}/health`,
      },
    ];
  },
};

module.exports = nextConfig;
