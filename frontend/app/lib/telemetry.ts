"use client";

/**
 * app/lib/telemetry.ts
 * ====================
 * Application Insights setup for the cv4 Next.js frontend.
 *
 * This initialises the Application Insights JavaScript SDK in the browser so
 * that client-side telemetry is collected and sent to Azure:
 *
 *   - Page views (every route the user visits)
 *   - JavaScript errors (uncaught exceptions in the browser)
 *   - AJAX/fetch calls the browser makes to the backend
 *   - Custom events (for example, a prediction being submitted)
 *
 * The connection string is read from the public environment variable
 * NEXT_PUBLIC_APPLICATIONINSIGHTS_CONNECTION_STRING. Note that, unlike the
 * backend, a browser connection string is necessarily visible to the client,
 * because the browser sends telemetry to Azure directly. This is normal and
 * expected for client-side monitoring.
 *
 * If the connection string is not set, initialisation is skipped and the app
 * runs normally without monitoring.
 */

import { ApplicationInsights } from "@microsoft/applicationinsights-web";

let appInsights: ApplicationInsights | null = null;

/**
 * Initialise Application Insights in the browser.
 *
 * Safe to call more than once; it only initialises on the first call. Does
 * nothing on the server (App Insights web SDK is browser-only) or when the
 * connection string is missing.
 */
export function initTelemetry(): void {
  // The web SDK only works in the browser, not during server-side rendering.
  if (typeof window === "undefined") {
    return;
  }

  // Already initialised, do not do it twice.
  if (appInsights) {
    return;
  }

  const connectionString =
    process.env.NEXT_PUBLIC_APPLICATIONINSIGHTS_CONNECTION_STRING;

  if (!connectionString) {
    // No connection string configured, skip monitoring quietly.
    console.warn(
      "NEXT_PUBLIC_APPLICATIONINSIGHTS_CONNECTION_STRING is not set. " +
        "Frontend monitoring is disabled.",
    );
    return;
  }

  appInsights = new ApplicationInsights({
    config: {
      connectionString,
      // We track route changes ourselves in TelemetryProvider so we can name
      // each page view by path; auto tracking would duplicate that.
      enableAutoRouteTracking: false,
      disableAjaxTracking: false,
      disableExceptionTracking: false,
    },
  });

  appInsights.loadAppInsights();
  // Send an initial page view, named by the URL path so each page shows
  // separately in monitoring instead of the shared document title.
  appInsights.trackPageView({ name: pageNameFromPath(window.location.pathname) });
}

export function trackEvent(
  name: string,
  properties?: Record<string, unknown>,
): void {
  if (!appInsights) {
    return;
  }
  appInsights.trackEvent({ name }, properties);
}

export function pageNameFromPath(path: string): string {
  if (!path || path === "/") {
    return "Home";
  }
  const slug = path.replace(/^\/+/, "").split("/")[0];
  return slug
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function trackPageViewByPath(path: string): void {
  if (!appInsights) {
    return;
  }
  appInsights.trackPageView({ name: pageNameFromPath(path) });
}