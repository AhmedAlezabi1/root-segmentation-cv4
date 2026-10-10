/**
 * app/lib/auth.ts
 * ===============
 * Client-side auth utilities for the cv4 JWT cookie system.
 *
 * Because tokens live in httpOnly cookies, JavaScript cannot read them
 * directly. This module handles:
 *
 *   - `authFetch`   — a drop-in replacement for `fetch` that automatically
 *                     attempts a token refresh on 401 and retries once.
 *   - `getMe`       — loads the current user from GET /api/auth/me.
 *   - `logout`      — calls POST /api/auth/logout to clear cookies.
 *
 * Usage
 * -----
 *   import { authFetch, getMe, logout } from "@/app/lib/auth";
 *
 *   // In a Server Component or client fetch:
 *   const res = await authFetch("/api/predict", { method: "POST", body: form });
 */

export interface AuthUser {
  id: number;
  email: string;
  is_active: boolean;
}

/**
 * Attempt to refresh the access token using the long-lived refresh cookie.
 * Returns true if the refresh succeeded, false otherwise.
 */
async function tryRefresh(): Promise<boolean> {
  try {
    const res = await fetch("/api/auth/refresh", { method: "POST", credentials: "include" });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * A fetch wrapper that transparently retries once after a 401 by calling
 * POST /api/auth/refresh. If the refresh also fails the original 401
 * Response is returned so the caller can redirect to /login.
 *
 * Always passes `credentials: "include"` so the browser sends the httpOnly
 * cookies on every request.
 *
 * @param input  URL string or Request — identical to fetch's first argument.
 * @param init   RequestInit options — credentials is always forced to "include".
 */
export async function authFetch(
  input: RequestInfo | URL,
  init: RequestInit = {}
): Promise<Response> {
  const opts: RequestInit = { ...init, credentials: "include" };
  const res = await fetch(input, opts);

  if (res.status !== 401) return res;

  // Access token expired — attempt silent refresh
  const refreshed = await tryRefresh();
  if (!refreshed) return res; // refresh also failed; caller handles redirect

  // Retry the original request with the fresh access token cookie
  return fetch(input, opts);
}

/**
 * Fetch the currently authenticated user from GET /api/auth/me.
 * Returns the user object, or null if not authenticated.
 * Times out after 5 seconds to prevent the page from hanging indefinitely.
 */
export async function getMe(): Promise<AuthUser | null> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    const res = await authFetch("/api/auth/me", { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) return null;
    return (await res.json()) as AuthUser;
  } catch {
    return null;
  }
}

/**
 * Call POST /api/auth/logout to expire both auth cookies server-side,
 * then navigate to /login.
 *
 * @param router  Optional Next.js router; if omitted, uses window.location.
 */
export async function logout(router?: { push: (path: string) => void }): Promise<void> {
  await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
  if (router) {
    router.push("/login");
  } else {
    window.location.href = "/login";
  }
}