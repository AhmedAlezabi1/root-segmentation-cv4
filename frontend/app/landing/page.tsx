"use client";

/**
 * app/landing/page.tsx
 * ====================
 * Public-facing landing page for the cv4 root segmentation tool.
 *
 * Accessible without authentication — acts as the entry point for new
 * visitors and unauthenticated users (page.tsx redirects here instead
 * of directly to /login).
 *
 * No auth check is performed; authenticated users may also visit this
 * page freely. Login and Register links are always shown.
 */

import Link from "next/link";

export default function LandingPage() {
  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .landing {
          min-height: 100vh;
          background: #f5f4f1;
          font-family: 'JetBrains Mono', 'Fira Mono', ui-monospace, monospace;
          color: #1a1a18;
        }

        /* ── Nav ─────────────────────────────────────────────────── */
        .nav {
          border-bottom: 1px solid #d6d4cc;
          padding: 1rem 2rem;
          display: flex;
          align-items: center;
          background: #faf9f6;
        }
        .nav-logo { font-size: 13px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #1a1a18; text-decoration: none; }
        .nav-org  { font-size: 11px; color: #8c8a82; margin-left: 10px; }
        .nav-actions { margin-left: auto; display: flex; gap: 8px; align-items: center; }

        .btn-outline {
          font-family: inherit; font-size: 11px; font-weight: 500;
          background: none; color: #6b6960;
          border: 1px solid #d6d4cc; border-radius: 3px;
          padding: 5px 14px; cursor: pointer; letter-spacing: 0.04em;
          text-decoration: none; display: inline-flex; align-items: center;
          transition: background 0.15s, color 0.15s, border-color 0.15s;
        }
        .btn-outline:hover { background: #eceae3; color: #1a1a18; border-color: #888680; }

        .btn-primary {
          font-family: inherit; font-size: 11px; font-weight: 500;
          background: #1a1a18; color: #f5f4f1;
          border: none; border-radius: 3px;
          padding: 6px 14px; cursor: pointer; letter-spacing: 0.04em;
          text-decoration: none; display: inline-flex; align-items: center;
          transition: background 0.15s;
        }
        .btn-primary:hover { background: #333330; }

        /* ── Hero ────────────────────────────────────────────────── */
        .hero {
          padding: 5rem 2rem 4.5rem;
          max-width: 680px;
          margin: 0 auto;
          text-align: center;
        }
        .hero-badge {
          display: inline-block;
          font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
          color: #2e6b10; background: #e8f5e0;
          border: 1px solid #8ec95a; border-radius: 3px;
          padding: 2px 10px; margin-bottom: 1.75rem;
        }
        .hero h1 {
          font-size: clamp(22px, 4vw, 34px);
          font-weight: 500; color: #1a1a18;
          letter-spacing: -0.02em; line-height: 1.25;
          margin-bottom: 1.25rem;
        }
        .hero p {
          font-size: 13px; color: #6b6960; line-height: 1.85;
          max-width: 500px; margin: 0 auto 2.25rem;
        }
        .hero-ctas { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }

        .btn-hero {
          font-family: inherit; font-size: 12px; font-weight: 500;
          background: #1a1a18; color: #f5f4f1;
          border: none; border-radius: 4px;
          padding: 10px 26px; cursor: pointer; letter-spacing: 0.04em;
          text-decoration: none; display: inline-flex; align-items: center; gap: 6px;
          transition: background 0.15s;
        }
        .btn-hero:hover { background: #333330; }

        .btn-hero-secondary {
          font-family: inherit; font-size: 12px; font-weight: 500;
          background: none; color: #6b6960;
          border: 1px solid #d6d4cc; border-radius: 4px;
          padding: 10px 26px; cursor: pointer; letter-spacing: 0.04em;
          text-decoration: none; display: inline-flex; align-items: center;
          transition: background 0.15s, color 0.15s;
        }
        .btn-hero-secondary:hover { background: #eceae3; color: #1a1a18; }

        /* ── Divider ─────────────────────────────────────────────── */
        .section-divider { border: none; border-top: 1px solid #d6d4cc; }

        /* ── Features ────────────────────────────────────────────── */
        .features-section {
          padding: 3.5rem 2rem;
          max-width: 940px;
          margin: 0 auto;
        }
        .section-label {
          font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
          color: #8c8a82; margin-bottom: 1.5rem; text-align: center;
        }
        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
          gap: 10px;
        }
        .feature-card {
          background: #faf9f6;
          border: 1px solid #d6d4cc;
          border-radius: 5px;
          padding: 20px 18px;
        }
        .feature-icon  { font-size: 16px; margin-bottom: 12px; color: #5a9e20; }
        .feature-title { font-size: 12px; font-weight: 500; color: #1a1a18; margin-bottom: 8px; letter-spacing: 0.04em; }
        .feature-desc  { font-size: 11px; color: #6b6960; line-height: 1.75; }

        /* ── How it works ────────────────────────────────────────── */
        .how-section {
          padding: 3.5rem 2rem;
          background: #faf9f6;
          border-top: 1px solid #d6d4cc;
          border-bottom: 1px solid #d6d4cc;
        }
        .how-inner { max-width: 940px; margin: 0 auto; }
        .steps-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
          gap: 16px;
          margin-top: 1.5rem;
        }
        .step { display: flex; gap: 14px; align-items: flex-start; }
        .step-num {
          width: 22px; height: 22px; border-radius: 50%;
          background: #1a1a18; color: #f5f4f1;
          font-size: 10px; font-weight: 500;
          display: flex; align-items: center; justify-content: center;
          flex-shrink: 0; margin-top: 1px;
        }
        .step-title { font-size: 12px; font-weight: 500; color: #1a1a18; margin-bottom: 6px; }
        .step-desc  { font-size: 11px; color: #6b6960; line-height: 1.75; }

        /* ── Footer ──────────────────────────────────────────────── */
        .footer {
          padding: 1.5rem 2rem;
          display: flex; align-items: center; justify-content: center;
          gap: 16px; flex-wrap: wrap;
        }
        .footer-text { font-size: 10px; color: #a09e96; }
        .footer-link { font-size: 10px; color: #6b6960; text-decoration: none; }
        .footer-link:hover { color: #1a1a18; }
      `}</style>

      <div className="landing">
        {/* Nav */}
        <nav className="nav">
          <Link href="/landing" className="nav-logo">Root Segmentation</Link>
          <span className="nav-org">cv4 · NPEC</span>
          <div className="nav-actions">
            <Link href="/register" className="btn-outline">Register</Link>
            <Link href="/login" className="btn-primary">Log in</Link>
          </div>
        </nav>

        {/* Hero */}
        <section className="hero">
          <span className="hero-badge">cv4 · NPEC Root Segmentation</span>
          <h1>Automated root tip detection<br />for petri dish imaging</h1>
          <p>
            Upload a petri dish image and get lane-by-lane root tip coordinates
            with confidence scores. Built for plant science researchers at NPEC.
          </p>
          <div className="hero-ctas">
            <Link href="/login" className="btn-hero">▶ Open the app</Link>
            <Link href="/register" className="btn-hero-secondary">Create an account</Link>
          </div>
        </section>

        <hr className="section-divider" />

        {/* Features */}
        <section className="features-section">
          <div className="section-label">What it does</div>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">⬡</div>
              <div className="feature-title">Lane detection</div>
              <div className="feature-desc">
                Automatically identifies and segments individual lanes in a petri
                dish image, detecting the presence or absence of a root in each.
              </div>
            </div>
            <div className="feature-card">
              <div className="feature-icon">◎</div>
              <div className="feature-title">Root tip localisation</div>
              <div className="feature-desc">
                Returns precise pixel coordinates for each detected root tip,
                overlaid as interactive markers on your original image.
              </div>
            </div>
            <div className="feature-card">
              <div className="feature-icon">◈</div>
              <div className="feature-title">Confidence scoring</div>
              <div className="feature-desc">
                Every detection includes a confidence score — colour-coded green,
                amber, or red — so you can spot uncertain results at a glance.
              </div>
            </div>
            <div className="feature-card">
              <div className="feature-icon">↺</div>
              <div className="feature-title">Feedback</div>
              <div className="feature-desc">
                Mark incorrect predictions and supply corrected coordinates. Images
                with feedback used to improve our model's performance.
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="footer">
          <span className="footer-text">cv4 · NPEC Root Segmentation</span>
          <span className="footer-text">·</span>
          <Link href="/login" className="footer-link">Log in</Link>
          <Link href="/register" className="footer-link">Register</Link>
        </footer>
      </div>
    </>
  );
}
