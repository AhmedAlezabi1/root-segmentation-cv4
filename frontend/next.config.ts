import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",

  // The /api/* proxy below buffers the request body (default cap 10MB).
  // The upload UI allows images up to 20MB, so raise the cap with headroom
  // for multipart boundary overhead — otherwise large uploads are truncated
  // and the backend receives a corrupt body.
  experimental: {
    proxyClientMaxBodySize: "25mb",
    // Inference can take >30s (the default 30000ms proxy timeout). Value is in
    // milliseconds — raise it past the client's 3-min abort window so the proxy
    // doesn't drop the connection mid-inference.
    proxyTimeout: 200_000,
  },

  // /api/* is handled by app/api/[...path]/route.ts at runtime,
  // which reads API_URL from the environment on each request.
  // This avoids baking the backend URL into the build.
};

export default nextConfig;
