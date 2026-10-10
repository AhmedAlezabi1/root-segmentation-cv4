"use client";

/**
 * app/reset-password/page.tsx
 * ============================
 * Step 2 of the password reset flow.
 *
 * Reads the reset token from the ?token= URL query parameter and submits
 * it alongside the new password to POST /api/auth/reset-password/confirm.
 * On success redirects to /login with a ?reset=1 flag so the login page
 * can show a confirmation banner.
 */

import { useState, FormEvent, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";

function validate(password: string, confirm: string): string | null {
  if (password.length < 8) return "Password must be at least 8 characters.";
  if (password.length > 128) return "Password must be 128 characters or fewer.";
  if (password !== confirm) return "Passwords do not match.";
  return null;
}

function ResetPasswordForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token") ?? "";

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (!token) {
    return (
      <div className="error-bar">
        No reset token found in the URL. Please request a new password reset link.
      </div>
    );
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);

    const validationError = validate(password, confirm);
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/auth/reset-password/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token,
          new_password: password,
          new_password_confirm: confirm,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail ?? "Password reset failed.");
        return;
      }

      router.push("/login?reset=1");
    } catch {
      setError("Could not reach the server. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {error && <div className="error-bar">{error}</div>}
      <form onSubmit={handleSubmit} noValidate>
        <div className="field">
          <label htmlFor="password">New password</label>
          <input
            id="password"
            type="password"
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
          />
          <div className="hint">Minimum 8 characters</div>
        </div>
        <div className="field">
          <label htmlFor="confirm">Confirm new password</label>
          <input
            id="confirm"
            type="password"
            autoComplete="new-password"
            required
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            placeholder="••••••••"
          />
        </div>
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? <><span className="spinner" />Saving…</> : "Set new password"}
        </button>
      </form>
    </>
  );
}

export default function ResetPasswordPage() {
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
          color-scheme: light;
        }
        .auth-card { width: 100%; max-width: 380px; }
        .auth-header { margin-bottom: 2rem; border-bottom: 1px solid #d6d4cc; padding-bottom: 1rem; }
        .auth-header h1 { font-size: 13px; font-weight: 500; color: #1a1a18; letter-spacing: 0.1em; text-transform: uppercase; }
        .auth-sub { font-size: 11px; color: #8c8a82; margin-top: 4px; }
        .field { margin-bottom: 1rem; }
        .field label { display: block; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: #8c8a82; margin-bottom: 5px; }
        .field input {
          width: 100%; font-family: inherit; font-size: 12px;
          background: #faf9f6; border: 1px solid #d6d4cc; border-radius: 4px;
          padding: 9px 12px; color: #1a1a18; outline: none; box-sizing: border-box;
        }
        .field input:focus { border-color: #888680; }
        .hint { font-size: 10px; color: #a09e96; margin-top: 4px; }
        .submit-btn {
          width: 100%; margin-top: 0.5rem; font-family: inherit; font-size: 12px;
          font-weight: 500; background: #1a1a18; color: #f5f4f1; border: none;
          border-radius: 4px; padding: 10px; cursor: pointer; letter-spacing: 0.04em;
          transition: background 0.15s;
        }
        .submit-btn:hover:not(:disabled) { background: #333330; }
        .submit-btn:disabled { background: #b8b6ae; cursor: not-allowed; }
        .error-bar { background: #fce8e8; border: 1px solid #e88a8a; border-radius: 4px; padding: 8px 12px; font-size: 11px; color: #9b2828; margin-bottom: 1rem; }
        .auth-footer { margin-top: 1.25rem; font-size: 11px; color: #8c8a82; text-align: center; }
        .auth-footer a { color: #1a1a18; text-decoration: underline; text-underline-offset: 2px; }
        .spinner { display: inline-block; width: 10px; height: 10px; border: 1.5px solid #888; border-top-color: #f5f4f1; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      <div className="auth-page">
        <div className="auth-card">
          <div className="auth-header">
            <h1>Set new password</h1>
            <div className="auth-sub">cv4 · NPEC Root Segmentation</div>
          </div>

          {/* useSearchParams requires Suspense in Next.js App Router */}
          <Suspense fallback={<div style={{ fontSize: 11, color: "#8c8a82" }}>Loading…</div>}>
            <ResetPasswordForm />
          </Suspense>

          <div className="auth-footer">
            <Link href="/login">Back to sign in</Link>
          </div>
        </div>
      </div>
    </>
  );
}
