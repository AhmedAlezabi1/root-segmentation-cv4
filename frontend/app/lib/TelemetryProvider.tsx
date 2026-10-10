"use client";
/**
 * app/lib/TelemetryProvider.tsx
 * =============================
 * A tiny client component that initialises Application Insights once on
 * first load, and sends a named page view on every route change so each
 * page shows separately in monitoring (instead of the shared document title).
 *
 * It is added near the top of the root layout so it runs on every page. It
 * renders nothing visible.
 */
import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { initTelemetry, trackPageViewByPath } from "./telemetry";

export default function TelemetryProvider() {
  const pathname = usePathname();

  useEffect(() => {
    initTelemetry();
  }, []);

  useEffect(() => {
    if (pathname) {
      trackPageViewByPath(pathname);
    }
  }, [pathname]);

  return null;
}