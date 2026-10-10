"use client";

/**
 * app/dashboard/page.tsx
 * ======================
 * Stakeholder dashboard for the cv4 root segmentation system.
 *
 * Shows: images processed per day, average confidence trend,
 * correction rate trend, drift status, and summary KPIs.
 *
 * Data is fetched from the /api/dashboard/* endpoints on mount
 * and every 5 minutes. A manual refresh button is also provided.
 * All fetches use authFetch so the JWT cookie is sent automatically.
 */

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authFetch, getMe, type AuthUser } from "@/app/lib/auth";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DayPoint {
  date: string;
  count?: number;
  avg_confidence_pct?: number | null;
  correction_rate_pct?: number | null;
}

interface Summary {
  total_predictions: number;
  total_feedback_lanes: number;
  roots_correct?: number;
  correction_rate_pct: number;
  avg_confidence_pct: number;
  model_version: string;
  cached_at: string;
}

interface DriftStatus {
  status: "green" | "yellow" | "red";
  explanation: string;
  rate_7d_pct: number | null;
  rate_30d_pct: number | null;
  cached_at: string;
}

interface DashboardData {
  summary: Summary | null;
  activity: DayPoint[];
  confidence: DayPoint[];
  correctionRate: DayPoint[];
  drift: DriftStatus | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const AUTO_REFRESH_MS = 5 * 60 * 1000; // 5 minutes

async function fetchDashboard(force = false): Promise<DashboardData> {
  const qs = force ? "?force_refresh=true" : "";
  const [summaryRes, activityRes, confidenceRes, correctionRes, driftRes] =
    await Promise.all([
      authFetch(`/api/dashboard/summary${qs}`),
      authFetch(`/api/dashboard/activity${qs}`),
      authFetch(`/api/dashboard/confidence${qs}`),
      authFetch(`/api/dashboard/correction-rate${qs}`),
      authFetch(`/api/dashboard/drift-status${qs}`),
    ]);

  const [summary, activity, confidence, correction, drift] = await Promise.all([
    summaryRes.ok ? summaryRes.json() : null,
    activityRes.ok ? activityRes.json() : { data: [] },
    confidenceRes.ok ? confidenceRes.json() : { data: [] },
    correctionRes.ok ? correctionRes.json() : { data: [] },
    driftRes.ok ? driftRes.json() : null,
  ]);

  return {
    summary,
    activity: activity.data ?? [],
    confidence: confidence.data ?? [],
    correctionRate: correction.data ?? [],
    drift,
  };
}

function fmt(n: number | null | undefined, suffix = ""): string {
  if (n == null) return "—";
  return `${n}${suffix}`;
}

function relativeTime(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

// ---------------------------------------------------------------------------
// Tiny bar chart (pure CSS, no library)
// ---------------------------------------------------------------------------

function BarChart({
  data,
  valueKey,
  color,
  suffix = "",
}: {
  data: DayPoint[];
  valueKey: "count" | "avg_confidence_pct" | "correction_rate_pct";
  color: string;
  suffix?: string;
}) {
  const values = data.map((d) => (d[valueKey] as number | null | undefined) ?? 0);
  const max = Math.max(...values, 1);

  return (
    <div className="chart-wrap">
      <div className="bar-chart">
        {data.map((d, i) => {
          const v = (d[valueKey] as number | null | undefined) ?? 0;
          const h = Math.round((v / max) * 100);
          const isLast7 = i >= data.length - 7;
          return (
            <div key={d.date} className="bar-col" title={`${d.date}: ${fmt(v, suffix)}`}>
              <div
                className="bar"
                style={{
                  height: `${h}%`,
                  background: color,
                  opacity: isLast7 ? 1 : 0.45,
                }}
              />
            </div>
          );
        })}
      </div>
      <div className="chart-axis">
        <span>{data[0]?.date?.slice(5)}</span>
        <span>last 7d darker</span>
        <span>{data[data.length - 1]?.date?.slice(5)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Drift badge
// ---------------------------------------------------------------------------

const DRIFT_CONFIG = {
  green:  { bg: "#e8f5e0", border: "#8ec95a", text: "#2e6b10", label: "Stable" },
  yellow: { bg: "#fef3dd", border: "#e8a020", text: "#7a4b0a", label: "Watch" },
  red:    { bg: "#fce8e8", border: "#e88a8a", text: "#9b2828", label: "Alert" },
};

function DriftBadge({ drift }: { drift: DriftStatus }) {
  const cfg = DRIFT_CONFIG[drift.status];
  return (
    <div
      className="drift-badge"
      style={{ background: cfg.bg, border: `1px solid ${cfg.border}`, color: cfg.text }}
    >
      <div className="drift-label">{cfg.label}</div>
      <div className="drift-explanation">{drift.explanation}</div>
      {drift.rate_7d_pct != null && drift.rate_30d_pct != null && (
        <div className="drift-rates">
          7d: {drift.rate_7d_pct}% · 30d: {drift.rate_30d_pct}%
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// KPI card
// ---------------------------------------------------------------------------

function KpiCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="kpi-card">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);

  const load = useCallback(async (force = false) => {
    try {
      const result = await fetchDashboard(force);
      setData(result);
      setLastRefresh(new Date().toISOString());
      setError(null);
    } catch {
      setError("Failed to load dashboard data. Please try again.");
    }
  }, []);

  // Auth check + initial load
  useEffect(() => {
    getMe().then((u) => {
      if (!u) { router.push("/login"); return; }
      setUser(u);
      load().finally(() => setLoading(false));
    });
  }, [router, load]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const id = setInterval(() => load(), AUTO_REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await load(true);
    setRefreshing(false);
  };

  // Download a ZIP of all images that have received feedback.
  //
  // Requires backend: GET /feedback/export
  //   Returns: application/zip — a ZIP archive named
  //   cv4-feedback-images-<date>.zip containing one image file per
  //   prediction that has been given feedback.
  const handleExport = async () => {
    setExportLoading(true);
    setError(null);
    try {
      const res = await authFetch("/api/feedback/export");
      if (res.status === 401) { router.push("/login"); return; }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(`Export failed: ${(body as { detail?: string }).detail ?? res.statusText}`);
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `cv4-feedback-images-${new Date().toISOString().slice(0, 10)}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError("Export failed. Could not reach the API.");
    } finally {
      setExportLoading(false);
    }
  };

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .dash-page {
          min-height: 100vh;
          background: #f5f4f1;
          font-family: 'JetBrains Mono', 'Fira Mono', ui-monospace, monospace;
          color: #1a1a18;
          color-scheme: light;
        }

        /* Header */
        .dash-header {
          border-bottom: 1px solid #d6d4cc;
          padding: 1rem 2rem;
          display: flex;
          align-items: center;
          gap: 16px;
          background: #faf9f6;
        }
        .dash-title { font-size: 13px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; }
        .dash-sub   { font-size: 11px; color: #8c8a82; }
        .header-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
        .user-tag { font-size: 10px; color: #8c8a82; }
        .refresh-btn {
          font-family: inherit; font-size: 11px; background: #1a1a18; color: #f5f4f1;
          border: none; border-radius: 3px; padding: 5px 12px; cursor: pointer;
          transition: background 0.15s; letter-spacing: 0.04em;
        }
        .refresh-btn:hover:not(:disabled) { background: #333330; }
        .refresh-btn:disabled { background: #b8b6ae; cursor: not-allowed; }
        .export-btn {
          font-family: inherit; font-size: 11px; background: none; color: #6b6960;
          border: 1px solid #d6d4cc; border-radius: 3px; padding: 5px 12px; cursor: pointer;
          transition: background 0.15s, color 0.15s; letter-spacing: 0.04em;
        }
        .export-btn:hover:not(:disabled) { background: #eceae3; color: #1a1a18; border-color: #888680; }
        .export-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .back-link { font-size: 11px; color: #8c8a82; text-decoration: none; }
        .back-link:hover { color: #1a1a18; }

        /* Body */
        .dash-body { padding: 2rem; max-width: 1100px; margin: 0 auto; }

        /* Status bar */
        .status-bar { font-size: 10px; color: #a09e96; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 8px; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: #8ec95a; display: inline-block; }

        /* KPI grid */
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 2rem; }
        .kpi-card { background: #faf9f6; border: 1px solid #d6d4cc; border-radius: 5px; padding: 14px 16px; }
        .kpi-value { font-size: 22px; font-weight: 500; color: #1a1a18; letter-spacing: -0.02em; margin-bottom: 4px; }
        .kpi-label { font-size: 10px; color: #8c8a82; text-transform: uppercase; letter-spacing: 0.08em; }
        .kpi-sub   { font-size: 10px; color: #a09e96; margin-top: 3px; }

        /* Charts grid */
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 2rem; }
        @media (max-width: 680px) { .charts-grid { grid-template-columns: 1fr; } }

        .chart-card { background: #faf9f6; border: 1px solid #d6d4cc; border-radius: 5px; padding: 16px; }
        .chart-title { font-size: 11px; font-weight: 500; color: #1a1a18; margin-bottom: 4px; letter-spacing: 0.04em; }
        .chart-desc  { font-size: 10px; color: #a09e96; margin-bottom: 12px; }

        .chart-wrap { width: 100%; }
        .bar-chart {
          display: flex; align-items: flex-end; gap: 2px;
          height: 80px; width: 100%;
        }
        .bar-col { flex: 1; display: flex; align-items: flex-end; height: 100%; }
        .bar { width: 100%; border-radius: 2px 2px 0 0; min-height: 2px; transition: height 0.3s ease; }
        .chart-axis { display: flex; justify-content: space-between; font-size: 9px; color: #a09e96; margin-top: 6px; }

        /* Drift */
        .drift-badge { border-radius: 5px; padding: 14px 16px; margin-bottom: 2rem; }
        .drift-label { font-size: 13px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
        .drift-explanation { font-size: 11px; line-height: 1.6; margin-bottom: 6px; }
        .drift-rates { font-size: 10px; opacity: 0.75; }

        /* Section label */
        .section-label { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: #8c8a82; margin-bottom: 10px; }

        /* Loading / error */
        .loading-state { display: flex; align-items: center; justify-content: center; min-height: 60vh; font-size: 11px; color: #8c8a82; }
        .error-bar { background: #fce8e8; border: 1px solid #e88a8a; border-radius: 4px; padding: 10px 14px; font-size: 11px; color: #9b2828; margin-bottom: 1.5rem; }

        .spinner { display: inline-block; width: 10px; height: 10px; border: 1.5px solid #888; border-top-color: #f5f4f1; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      <div className="dash-page">
        {/* Header */}
        <div className="dash-header">
          <div>
            <div className="dash-title">System Dashboard</div>
            <div className="dash-sub">cv4 · NPEC Root Segmentation</div>
          </div>
          <div className="header-right">
            {user && <span className="user-tag">⬡ {user.email}</span>}
            <Link href="/" className="back-link">← Back to app</Link>
            <button
              className="export-btn"
              onClick={handleExport}
              disabled={exportLoading}
              title="Download a ZIP of all images that have received feedback"
            >
              {exportLoading ? <><span className="spinner" />Exporting…</> : "↓ Export images"}
            </button>
            <button
              className="refresh-btn"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              {refreshing ? <><span className="spinner" />Refreshing…</> : "↺ Refresh"}
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="dash-body">
          {loading ? (
            <div className="loading-state">Loading dashboard…</div>
          ) : (
            <>
              {/* Status bar */}
              {lastRefresh && (
                <div className="status-bar">
                  <span className="status-dot" />
                  Data updated {relativeTime(lastRefresh)} · auto-refreshes every 5 min
                </div>
              )}

              {error && <div className="error-bar">{error}</div>}

              {/* KPIs */}
              {data?.summary && (
                <>
                  <div className="section-label">Overview</div>
                  <div className="kpi-grid" style={{ marginBottom: "2rem" }}>
                    <KpiCard
                      label="Images processed"
                      value={data.summary.total_predictions.toLocaleString()}
                      sub="all time"
                    />
                    <KpiCard
                      label="Avg confidence"
                      value={fmt(data.summary.avg_confidence_pct, "%")}
                      sub="all lanes with root"
                    />
                    <KpiCard
                      label="Roots correct"
                      value={data.summary.total_feedback_lanes != null && data.summary.correction_rate_pct != null
                        ? Math.round(data.summary.total_feedback_lanes * (1 - data.summary.correction_rate_pct / 100)).toLocaleString()
                        : "—"}
                      sub={`of ${data.summary.total_feedback_lanes.toLocaleString()} roots reviewed`}
                    />
                    <KpiCard
                      label="Correct"
                      value={data.summary.correction_rate_pct != null ? `${100 - data.summary.correction_rate_pct}%` : "—"}
                      sub="of all feedback roots"
                    />
                    <KpiCard
                      label="Model version"
                      value={data.summary.model_version}
                      sub="currently deployed"
                    />
                  </div>
                </>
              )}



              {/* Charts */}
              <div className="section-label">Trends — last 30 days</div>
              <div className="charts-grid">
                <div className="chart-card">
                  <div className="chart-title">Images processed per day</div>
                  <div className="chart-desc">Number of predictions run each day</div>
                  <BarChart
                    data={data?.activity ?? []}
                    valueKey="count"
                    color="#5a9e20"
                  />
                </div>

                <div className="chart-card">
                  <div className="chart-title">Average confidence</div>
                  <div className="chart-desc">Model certainty at detected root tip (%)</div>
                  <BarChart
                    data={data?.confidence ?? []}
                    valueKey="avg_confidence_pct"
                    color="#4a7eb5"
                    suffix="%"
                  />
                </div>

                <div className="chart-card">
                  <div className="chart-title">Correct</div>
                  <div className="chart-desc">% of feedback roots marked correct per day</div>
                  <BarChart
                    data={data?.correctionRate?.map(d => ({
                      ...d,
                      correction_rate_pct: d.correction_rate_pct != null ? 100 - d.correction_rate_pct : null,
                    })) ?? []}
                    valueKey="correction_rate_pct"
                    color="#5a9e20"
                    suffix="%"
                  />
                </div>

                <div className="chart-card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                  <div>
                    <div className="chart-title">How to read these charts</div>
                    <div className="chart-desc" style={{ lineHeight: 1.8 }}>
                      <strong>Confidence</strong> — higher is better. A falling trend may
                      indicate harder images or model drift.<br /><br />
                      <strong>Correct</strong> — higher is better. A falling trend
                      means users are correcting more predictions, which may signal drift.<br /><br />
                      <strong>Darker bars</strong> represent the last 7 days.
                    </div>
                  </div>
                  {data?.summary?.cached_at && (
                    <div style={{ fontSize: 9, color: "#a09e96", marginTop: 12 }}>
                      Cache computed {relativeTime(data.summary.cached_at)} · TTL 5 min
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
