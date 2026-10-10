"use client";

/**
 * app/login/page.tsx
 * ==================
 * Login page for the cv4 root segmentation tool.
 *
 * Submits email + password to POST /api/auth/login.
 * On success the backend sets httpOnly cookies and returns the user object;
 * this page then navigates to the home page.
 */

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

interface ApiError {
  detail: string;
}

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        credentials: "include",          // ensure cookies are accepted
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
      });

      if (res.ok) {
        router.push("/");
        return;
      }

      const data: ApiError = await res.json().catch(() => ({ detail: "Login failed." }));
      setError(data.detail ?? "Login failed.");
    } catch {
      setError("Could not reach the server. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{`
        .auth-page {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 2rem 1rem;
          background: var(--background, #f5f4f1);
        }
        .auth-card {
          width: 100%;
          max-width: 380px;
        }
        .auth-header {
          margin-bottom: 2rem;
          border-bottom: 1px solid #d6d4cc;
          padding-bottom: 1rem;
        }
        .auth-header h1 {
          font-size: 13px;
          font-weight: 500;
          color: #1a1a18;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
        .auth-header .auth-sub {
          font-size: 11px;
          color: #8c8a82;
          margin-top: 4px;
        }
        .field {
          margin-bottom: 1rem;
        }
        .field label {
          display: block;
          font-size: 10px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #8c8a82;
          margin-bottom: 5px;
        }
        .field input {
          width: 100%;
          font-family: inherit;
          font-size: 12px;
          background: #faf9f6;
          border: 1px solid #d6d4cc;
          border-radius: 4px;
          padding: 9px 12px;
          color: #1a1a18;
          outline: none;
          box-sizing: border-box;
        }
        .field input:focus {
          border-color: #888680;
        }
        .submit-btn {
          width: 100%;
          margin-top: 0.5rem;
          font-family: inherit;
          font-size: 12px;
          font-weight: 500;
          background: #1a1a18;
          color: #f5f4f1;
          border: none;
          border-radius: 4px;
          padding: 10px;
          cursor: pointer;
          letter-spacing: 0.04em;
          transition: background 0.15s;
        }
        .submit-btn:hover:not(:disabled) { background: #333330; }
        .submit-btn:disabled { background: #b8b6ae; cursor: not-allowed; }
        .error-bar {
          background: #fce8e8;
          border: 1px solid #e88a8a;
          border-radius: 4px;
          padding: 8px 12px;
          font-size: 11px;
          color: #9b2828;
          margin-bottom: 1rem;
        }
        .auth-footer {
          margin-top: 1.25rem;
          font-size: 11px;
          color: #8c8a82;
          text-align: center;
        }
        .auth-footer a {
          color: #1a1a18;
          text-decoration: underline;
          text-underline-offset: 2px;
        }
        .spinner {
          display: inline-block;
          width: 10px;
          height: 10px;
          border: 1.5px solid #888;
          border-top-color: #f5f4f1;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
          vertical-align: middle;
          margin-right: 6px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      <div className="auth-page">
        <div className="auth-card">
          <div className="auth-header">
            <h1>Sign in</h1>
            <div className="auth-sub">cv4 · NPEC Root Segmentation</div>
          </div>

          {error && <div className="error-bar">{error}</div>}

          <form onSubmit={handleSubmit} noValidate>
            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
              />
            </div>

            <div className="field">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
              />
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? <><span className="spinner" />Signing in…</> : "Sign in"}
            </button>
          </form>

          <div className="auth-footer">
            <Link href="/register">Create account</Link>
            {" · "}
            <Link href="/forgot-password">Forgot password?</Link>
          </div>
        </div>
      </div>
    </>
  );
}
