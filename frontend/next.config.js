/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: 'https://backend-production-4622.up.railway.app',
  },
  rewrites: async () => {
    return [
      {
        source: '/api/:path*',
        destination: 'https://backend-production-4622.up.railway.app/api/:path*',
      },
      {
        source: '/ws/:path*',
        destination: 'https://backend-production-4622.up.railway.app/ws/:path*',
      },
      {
        source: '/health',
        destination: 'https://backend-production-4622.up.railway.app/health',
      },
    ];
  },
};

module.exports = nextConfig;
