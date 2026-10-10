"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authFetch, getMe, logout, type AuthUser } from "@/app/lib/auth";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RootTipPx {
  x: number;
  y: number;
}

interface RootResult {
  lane: number;
  has_root: boolean;
  root_tip_px: RootTipPx | null;
  confidence: number | null;
}

interface CropBox {
  x: number;
  y: number;
  side: number;
}

interface PredictResponse {
  prediction_id: number;
  roots: RootResult[];
  processing_time_s: number;
  crop: CropBox;
}

interface LaneFeedbackRow {
  lane: number;
  is_correct: boolean;
  correct_tip_x: string;
  correct_tip_y: string;
  comment: string;
}

type HealthStatus = "checking" | "ready" | "offline";

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB
const INFERENCE_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes (CPU inference can be slow on cold start)
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/tiff"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function confPct(c: number | null): string {
  return c != null ? (c * 100).toFixed(1) + "%" : "—";
}

function tipStr(t: RootTipPx | null): string {
  return t ? `${t.x}, ${t.y}` : "—";
}

// Confidence → color: green (≥90%) → amber (70–90%) → red (<70%)
function confColor(c: number | null): string {
  if (c == null) return "#b8b6ae";
  if (c >= 0.9)  return "#5a9e20";
  if (c >= 0.7)  return "#e8a020";
  return "#c0392b";
}

// Overlay: image + SVG markers + hover tooltips
function PredictionOverlay({
  previewUrl,
  roots,
  crop,
}: {
  previewUrl: string;
  roots: RootResult[];
  crop: CropBox;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number; nw: number; nh: number } | null>(null);
  const [tooltip, setTooltip] = useState<RootResult | null>(null);

  const measure = useCallback(() => {
    const el = imgRef.current;
    if (!el) return;
    setImgSize({ w: el.naturalWidth, h: el.naturalHeight, nw: el.clientWidth, nh: el.clientHeight });
  }, []);

  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    if (el.complete) measure();
    el.addEventListener("load", measure);
    window.addEventListener("resize", measure);
    return () => {
      el.removeEventListener("load", measure);
      window.removeEventListener("resize", measure);
    };
  }, [measure]);

  const MARKER_R = 7;

  return (
    <div style={{ position: "relative", display: "block", width: "100%" }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={previewUrl}
        alt="Prediction overlay"
        onLoad={measure}
        style={{ width: "100%", display: "block", borderRadius: 4, border: "1px solid #d6d4cc" }}
      />
      {imgSize && (
        <svg
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible" }}
          viewBox={`0 0 ${imgSize.nw} ${imgSize.nh}`}
          onMouseLeave={() => setTooltip(null)}
        >
          {roots.filter((r) => r.has_root && r.root_tip_px).map((r) => {
            // root_tip_px is in crop space — offset by crop origin to get
            // full-image coordinates, then scale to displayed image size.
            const scaleX = imgSize.nw / imgSize.w;
            const scaleY = imgSize.nh / imgSize.h;
            const cx = (crop.x + r.root_tip_px!.x) * scaleX;
            const cy = (crop.y + r.root_tip_px!.y) * scaleY;
            const color = confColor(r.confidence);
            return (
              <g key={r.lane} onMouseEnter={() => setTooltip(r)} style={{ cursor: "pointer" }}>
                <circle cx={cx} cy={cy} r={MARKER_R + 3} fill="none" stroke={color} strokeWidth={1.5} opacity={0.4} />
                <circle cx={cx} cy={cy} r={MARKER_R} fill={color} stroke="#fff" strokeWidth={1.5} opacity={0.92} />
                <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
                  fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="600" fill="#fff">
                  {r.lane}
                </text>
              </g>
            );
          })}

          {/* Tooltip */}
          {tooltip && tooltip.root_tip_px && (() => {
            const scaleX = imgSize.nw / imgSize.w;
            const scaleY = imgSize.nh / imgSize.h;
            const cx = (crop.x + tooltip.root_tip_px.x) * scaleX;
            const cy = (crop.y + tooltip.root_tip_px.y) * scaleY;
            const TW = 130, TH = 52, PAD = 6;
            const tx = Math.min(cx + 14, imgSize.nw - TW - PAD);
            const ty = Math.max(cy - TH / 2, PAD);
            return (
              <g pointerEvents="none">
                <rect x={tx} y={ty} width={TW} height={TH} rx={4} fill="#1a1a18" opacity={0.88} />
                <text x={tx + 8} y={ty + 14} fontSize={10} fontFamily="'JetBrains Mono', monospace" fill="#f5f4f1" fontWeight="600">
                  Lane {tooltip.lane}
                </text>
                <text x={tx + 8} y={ty + 28} fontSize={9} fontFamily="'JetBrains Mono', monospace" fill="#a09e96">
                  tip: ({tooltip.root_tip_px.x}, {tooltip.root_tip_px.y})
                </text>
                <text x={tx + 8} y={ty + 42} fontSize={9} fontFamily="'JetBrains Mono', monospace" fill={confColor(tooltip.confidence)}>
                  conf: {confPct(tooltip.confidence)}
                </text>
              </g>
            );
          })()}
        </svg>
      )}
    </div>
  );
}

// Summary panel — listed alongside the overlay image
function SummaryPanel({ roots }: { roots: RootResult[] }) {
  const detected = roots.filter((r) => r.has_root).length;
  return (
    <div className="summary-panel">
      <div className="summary-header">{detected} / {roots.length} roots detected</div>
      <div className="summary-rows">
        {roots.map((r) => (
          <div key={r.lane} className={`summary-row ${r.has_root ? "" : "summary-row-none"}`}>
            <span className="summary-lane">
              <span className="dot" style={{ background: confColor(r.confidence), width: 7, height: 7, borderRadius: "50%", display: "inline-block", flexShrink: 0 }} />
              Lane {r.lane}
            </span>
            <span className="summary-tip">{tipStr(r.root_tip_px)}</span>
            <span className="summary-conf" style={{ color: r.has_root ? confColor(r.confidence) : "#b8b6ae" }}>
              {confPct(r.confidence)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: HealthStatus }) {
  const map: Record<HealthStatus, { label: string; cls: string }> = {
    checking: { label: "checking…", cls: "badge-loading" },
    ready:    { label: "model ready", cls: "badge-ready" },
    offline:  { label: "offline", cls: "badge-offline" },
  };
  const { label, cls } = map[status];
  return <span className={`badge ${cls}`}>{label}</span>;
}

function LaneCard({ root }: { root: RootResult }) {
  return (
    <div className={`lane-card ${root.has_root ? "has-root" : "no-root"}`}>
      <div className="lane-num">Lane {root.lane}</div>
      <div className="lane-detect">
        <span className={`dot ${root.has_root ? "dot-yes" : "dot-no"}`} />
        <span>{root.has_root ? "detected" : "none"}</span>
      </div>
      <div className="lane-stat">
        <strong>{tipStr(root.root_tip_px)}</strong>
        <span>conf: {confPct(root.confidence)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Home() {
  const router = useRouter();

  const [health, setHealth] = useState<HealthStatus>("checking");
  const [user, setUser] = useState<AuthUser | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [feedbackRows, setFeedbackRows] = useState<LaneFeedbackRow[]>([]);

  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [fbLoading, setFbLoading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load current user and check API health in parallel on mount.
  // If getMe returns null the access token is missing or expired — redirect
  // to /login so the user can authenticate.
  useEffect(() => {
    Promise.all([
      fetch("/api/health", { credentials: "include" })
        .then((r) => setHealth(r.ok ? "ready" : "offline"))
        .catch(() => setHealth("offline")),
      getMe().then((u) => {
        if (!u) {
          router.push("/landing"); // landing page for unauthenticated visitors
        } else {
          setUser(u);
        }
      }),
    ]);
  }, [router]);

  const handleFile = useCallback((f: File) => {
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError(`Invalid file type "${f.type}". Please upload a PNG, JPEG, or TIFF image.`);
      return;
    }
    if (f.size > MAX_FILE_SIZE) {
      setError(`File is too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Maximum allowed size is 20 MB.`);
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setPrediction(null);
    setError(null);
    setSuccess(null);
  }, []);

  const clearFile = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    setSuccess(null);
    setFeedbackRows([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [previewUrl]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const buildFeedbackRows = (roots: RootResult[]): LaneFeedbackRow[] =>
    roots.map((r) => ({
      lane: r.lane,
      is_correct: r.has_root,
      correct_tip_x: r.root_tip_px ? String(r.root_tip_px.x) : "",
      correct_tip_y: r.root_tip_px ? String(r.root_tip_px.y) : "",
      comment: "",
    }));

  const updateFbRow = (
    lane: number,
    field: keyof LaneFeedbackRow,
    value: string | boolean
  ) => {
    setFeedbackRows((rows) =>
      rows.map((r) => (r.lane === lane ? { ...r, [field]: value } : r))
    );
  };

  // Run inference
  const runInference = async () => {
    if (!file || !user) return;
    setLoading(true);
    setError(null);
    setSuccess(null);
    setPrediction(null);

    const form = new FormData();
    form.append("image", file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), INFERENCE_TIMEOUT_MS);

    try {
      // authFetch sends the httpOnly cookie automatically and silently retries
      // once after a 401 by calling POST /api/auth/refresh.
      const res = await authFetch("/api/predict", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      if (res.status === 401) {
        router.push("/login");
        return;
      }
      const data = await res.json();
      if (!res.ok) {
        setError(`${res.status}: ${data.detail ?? "Inference failed"}`);
        return;
      }
      setPrediction(data);
      setFeedbackRows(buildFeedbackRows(data.roots));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Request timed out after 10 minutes. The image may be too large or the server is overloaded.");
      } else {
        setError("Could not reach the API. Is the server running?");
      }
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  // Submit feedback
  const submitFeedback = async () => {
    if (!prediction) return;

    // Validate corrections
    const invalid = feedbackRows.find(
      (r) =>
        !r.is_correct &&
        (r.correct_tip_x.trim() === "" || r.correct_tip_y.trim() === "")
    );
    if (invalid) {
      setError(
        `Lane ${invalid.lane}: corrected tip coordinates are required when marking as incorrect.`
      );
      return;
    }

    setFbLoading(true);
    setError(null);
    setSuccess(null);

    const lanes = feedbackRows.map((r) => ({
      lane: r.lane,
      is_correct: r.is_correct,
      predicted_tip_x: prediction.roots[r.lane].root_tip_px?.x ?? null,
      predicted_tip_y: prediction.roots[r.lane].root_tip_px?.y ?? null,
      correct_tip_x: !r.is_correct ? parseInt(r.correct_tip_x) : null,
      correct_tip_y: !r.is_correct ? parseInt(r.correct_tip_y) : null,
      comment: r.comment || null,
    }));

    try {
      const res = await authFetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_id: prediction.prediction_id,
          lanes,
        }),
      });
      if (res.status === 401) {
        router.push("/login");
        return;
      }
      const data = await res.json();
      if (!res.ok) {
        setError(`${res.status}: ${data.detail ?? "Feedback submission failed"}`);
        return;
      }
      const ids = data.submitted
        .map((s: { lane: number; feedback_id: number }) => `L${s.lane}:#${s.feedback_id}`)
        .join(" · ");
      setSuccess(`Feedback saved — ${ids}`);

      // Store the image server-side so it is available for export from the dashboard.
      // Fire-and-forget — a failure here does not affect the feedback confirmation.
      //
      // Requires backend: POST /feedback/{prediction_id}/image
      //   Accepts: multipart/form-data with field "image"
      //   Stores the image associated with this prediction_id.
      if (file) {
        const imgForm = new FormData();
        imgForm.append("image", file);
        authFetch(`/api/feedback/${prediction.prediction_id}/image`, {
          method: "POST",
          body: imgForm,
        }).catch(() => { /* silent — export is supplementary */ });
      }
    } catch {
      setError("Could not reach the API while submitting feedback.");
    } finally {
      setFbLoading(false);
    }
  };

  const canRun = !!file && !!user && !loading;

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f5f4f1; font-family: 'JetBrains Mono', 'Fira Mono', ui-monospace, monospace; }

        .page { min-height: 100vh; padding: 2.5rem 1.5rem; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 720px; }

        .header { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; border-bottom: 1px solid #d6d4cc; padding-bottom: 1rem; }
        .header-left { display: flex; align-items: baseline; gap: 12px; }
        .header h1 { font-size: 13px; font-weight: 500; color: #1a1a18; letter-spacing: 0.1em; text-transform: uppercase; }
        .header .org { font-size: 11px; color: #8c8a82; }
        .header-right { display: flex; align-items: center; gap: 10px; margin-left: auto; }
        .badge { font-size: 10px; padding: 2px 8px; border-radius: 3px; border: 1px solid; }
        .badge-ready   { background: #e8f5e0; color: #2e6b10; border-color: #8ec95a; }
        .badge-offline { background: #fce8e8; color: #9b2828; border-color: #e88a8a; }
        .badge-loading { background: #fef3dd; color: #7a4b0a; border-color: #e8a020; }

        .header-user { display: flex; align-items: center; gap: 8px; }
        .header-email { font-size: 11px; color: #6b6960; background: #eceae3; border: 1px solid #d6d4cc; border-radius: 3px; padding: 4px 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
        .header-nav-btn {
          font-family: inherit; font-size: 11px; font-weight: 500;
          background: #1a1a18; color: #f5f4f1;
          border: none; border-radius: 3px;
          padding: 5px 12px; cursor: pointer;
          letter-spacing: 0.04em; white-space: nowrap;
          text-decoration: none; display: inline-flex; align-items: center;
          transition: background 0.15s;
        }
        .header-nav-btn:hover { background: #333330; }
        .header-nav-btn.secondary {
          background: none; color: #6b6960;
          border: 1px solid #d6d4cc;
        }
        .header-nav-btn.secondary:hover { background: #eceae3; color: #1a1a18; border-color: #888680; }

        .upload-zone {
          border: 1px dashed #b8b6ae; border-radius: 6px; padding: 2rem 1.5rem;
          text-align: center; cursor: pointer; position: relative;
          background: #faf9f6; transition: background 0.15s, border-color 0.15s;
          margin-bottom: 1rem;
        }
        .upload-zone:hover, .upload-zone.dragging {
          background: #f0efe9; border-color: #888680; border-style: solid;
        }
        .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; }
        .upload-icon { font-size: 22px; color: #a09e96; margin-bottom: 8px; }
        .upload-label { font-size: 12px; color: #6b6960; }
        .upload-hint  { font-size: 10px; color: #a09e96; margin-top: 4px; }
        .preview-img  { max-height: 160px; max-width: 100%; border-radius: 4px; border: 1px solid #d6d4cc; margin-top: 12px; }
        .filename-pill {
          display: inline-block; font-size: 10px; background: #eceae3;
          border: 1px solid #d6d4cc; border-radius: 3px; padding: 2px 8px; color: #5e5c55;
          max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
          margin-bottom: 1rem;
        }
        .preview-wrapper { position: relative; display: inline-block; margin-top: 12px; }
        .clear-btn {
          position: absolute; top: -8px; right: -8px;
          width: 20px; height: 20px; border-radius: 50%;
          background: #1a1a18; color: #f5f4f1;
          border: none; cursor: pointer; font-size: 11px; line-height: 1;
          display: flex; align-items: center; justify-content: center;
          transition: background 0.15s; z-index: 1;
        }
        .clear-btn:hover { background: #9b2828; }

        .api-row { display: flex; gap: 8px; margin-bottom: 1.5rem; align-items: center; justify-content: center; }

        .run-btn, .submit-btn {
          font-family: inherit; font-size: 12px; font-weight: 500;
          background: #1a1a18; color: #f5f4f1; border: none; border-radius: 4px;
          padding: 8px 20px; cursor: pointer; letter-spacing: 0.04em;
          transition: background 0.15s; white-space: nowrap;
        }
        .run-btn:hover, .submit-btn:hover { background: #333330; }
        .run-btn:disabled, .submit-btn:disabled { background: #b8b6ae; cursor: not-allowed; }

        .section-label { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: #8c8a82; margin-bottom: 10px; }

        .viz-layout { display: grid; grid-template-columns: 1fr 200px; gap: 12px; align-items: start; margin-bottom: 1rem; }
        @media (max-width: 580px) { .viz-layout { grid-template-columns: 1fr; } }
        .conf-legend { display: flex; gap: 12px; font-size: 10px; color: #8c8a82; align-items: center; margin-bottom: 2rem; }
        .conf-legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

        .summary-panel { background: #faf9f6; border: 1px solid #d6d4cc; border-radius: 6px; overflow: hidden; }
        .summary-header { font-size: 10px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: #8c8a82; padding: 8px 12px; border-bottom: 1px solid #e0ddd6; }
        .summary-rows { padding: 4px 0; }
        .summary-row { display: flex; align-items: center; gap: 6px; padding: 5px 12px; font-size: 11px; }
        .summary-row-none { opacity: 0.5; }
        .summary-lane { display: flex; align-items: center; gap: 6px; flex: 1; color: #1a1a18; }
        .summary-tip  { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6b6960; flex: 1; text-align: right; white-space: nowrap; }
        .summary-conf { font-family: 'JetBrains Mono', monospace; font-size: 10px; width: 44px; text-align: right; flex-shrink: 0; }

        .error-bar   { background: #fce8e8; border: 1px solid #e88a8a; border-radius: 4px; padding: 8px 12px; font-size: 11px; color: #9b2828; margin-bottom: 12px; }
        .success-bar { background: #e8f5e0; border: 1px solid #8ec95a; border-radius: 4px; padding: 8px 12px; font-size: 11px; color: #2e6b10; margin-bottom: 12px; }

        .result-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
        .pred-id   { font-size: 11px; background: #eceae3; border: 1px solid #d6d4cc; border-radius: 3px; padding: 2px 8px; color: #5e5c55; }
        .proc-time { font-size: 11px; color: #a09e96; margin-left: auto; }

        .lanes-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-bottom: 2rem; }
        .lane-card { background: #faf9f6; border: 1px solid #d6d4cc; border-radius: 5px; padding: 10px 8px; }
        .lane-card.has-root { border-color: #8ec95a; }
        .lane-card.no-root  { opacity: 0.55; }
        .lane-num  { font-size: 10px; color: #a09e96; margin-bottom: 6px; }
        .lane-detect { display: flex; align-items: center; gap: 5px; font-size: 10px; color: #6b6960; margin-bottom: 6px; }
        .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
        .dot-yes { background: #5a9e20; }
        .dot-no  { background: #b8b6ae; }
        .lane-stat { font-size: 10px; color: #6b6960; line-height: 1.7; }
        .lane-stat strong { display: block; font-size: 11px; color: #1a1a18; font-weight: 500; }

        .feedback-box { background: #faf9f6; border: 1px solid #d6d4cc; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 2rem; }
        .fb-table { width: 100%; border-collapse: collapse; font-size: 11px; margin: 8px 0 12px; }
        .fb-table th { text-align: left; color: #a09e96; font-weight: 400; font-size: 10px; letter-spacing: 0.07em; text-transform: uppercase; padding: 0 6px 8px; border-bottom: 1px solid #e0ddd6; }
        .fb-table td { padding: 6px; vertical-align: middle; border-bottom: 1px solid #e0ddd6; color: #1a1a18; }
        .fb-table tr:last-child td { border-bottom: none; }
        .fb-table input[type="number"] {
          width: 60px; font-family: inherit; font-size: 10px;
          background: #f0efe9; border: 1px solid #d6d4cc; border-radius: 3px; padding: 3px 5px; color: #1a1a18;
        }
        .fb-table input[type="number"]:disabled { opacity: 0.4; }
        .fb-table input[type="checkbox"] { width: 14px; height: 14px; cursor: pointer; accent-color: #5a9e20; }
        .fb-table textarea {
          font-family: inherit; font-size: 10px; width: 100%; resize: none;
          background: #f0efe9; border: 1px solid #d6d4cc; border-radius: 3px; padding: 3px 6px; color: #1a1a18;
        }
        .coord-pair { display: flex; gap: 4px; align-items: center; }
        .coord-lbl  { font-size: 10px; color: #a09e96; }
        .submit-row { display: flex; align-items: center; justify-content: flex-end; gap: 12px; }
        .fb-note { font-size: 10px; color: #8c8a82; }

        .spinner {
          display: inline-block; width: 11px; height: 11px;
          border: 1.5px solid #d6d4cc; border-top-color: #6b6960;
          border-radius: 50%; animation: spin 0.7s linear infinite;
          vertical-align: middle; margin-right: 6px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      <div className="page">
        <div className="container">
          {/* Header */}
          <div className="header">
            <div className="header-left">
              <h1>Root Segmentation</h1>
              <span className="org">cv4 · NPEC</span>
            </div>
            <div className="header-right">
              <StatusBadge status={health} />
              {user && (
                <div className="header-user">
                  <span className="header-email">⬡ {user.email}</span>
                  <Link href="/dashboard" className="header-nav-btn secondary">Dashboard</Link>
                  <button
                    className="header-nav-btn"
                    onClick={() => logout(router)}
                    title="Sign out"
                  >
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Upload zone */}
          <div
            className={`upload-zone ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/tiff"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              style={{ display: "none" }}
            />
            <div className="upload-icon">↑</div>
            <div className="upload-label">Drag & drop petri dish image, or click to browse</div>
            <div className="upload-hint">PNG · JPEG · TIFF · max 20 MB</div>
            {previewUrl && (
              <div className="preview-wrapper" onClick={(e) => e.stopPropagation()}>
                <button className="clear-btn" onClick={clearFile} title="Remove image">✕</button>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewUrl} alt="Preview" className="preview-img" style={{ margin: 0 }} />
              </div>
            )}
          </div>

          {file && <div className="filename-pill">{file.name}</div>}

          {/* Run button */}
          <div className="api-row">
            <button className="run-btn" onClick={runInference} disabled={!canRun}>
              {loading ? <><span className="spinner" />Running…</> : "▶ Run inference"}
            </button>
          </div>

          {/* Feedback / error bars */}
          {error   && <div className="error-bar">{error}</div>}
          {success && <div className="success-bar">{success}</div>}

          {/* Results */}
          {prediction && (
            <>
              <div className="section-label" style={{ marginBottom: 10 }}>Prediction results</div>
              <div className="result-header">
                <span className="pred-id">prediction #{prediction.prediction_id}</span>
                <span className="proc-time">{prediction.processing_time_s.toFixed(2)}s</span>
              </div>

              {/* Visualization overlay */}
              <div className="viz-layout">
                <PredictionOverlay previewUrl={previewUrl!} roots={prediction.roots} crop={prediction.crop} />
                <SummaryPanel roots={prediction.roots} />
              </div>
              <div className="conf-legend">
                <span className="conf-legend-dot" style={{ background: "#5a9e20" }} /><span>≥90%</span>
                <span className="conf-legend-dot" style={{ background: "#e8a020" }} /><span>70–90%</span>
                <span className="conf-legend-dot" style={{ background: "#c0392b" }} /><span>&lt;70%</span>
              </div>

              <div className="lanes-grid">
                {prediction.roots.map((r) => <LaneCard key={r.lane} root={r} />)}
              </div>

              {/* Inline feedback */}
              <div className="section-label">Feedback</div>
              <div className="feedback-box">
                <table className="fb-table">
                  <thead>
                    <tr>
                      <th>Lane</th>
                      <th>Model output</th>
                      <th>Correct?</th>
                      <th>Corrected tip (x, y)</th>
                      <th>Comment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {feedbackRows.map((row) => {
                      const root = prediction.roots[row.lane];
                      return (
                        <tr key={row.lane}>
                          <td>{row.lane}</td>
                          <td style={{ color: "#6b6960" }}>
                            {root.has_root
                              ? `(${root.root_tip_px!.x}, ${root.root_tip_px!.y}) · ${confPct(root.confidence)}`
                              : "no root"}
                          </td>
                          <td style={{ textAlign: "center" }}>
                            <input
                              type="checkbox"
                              checked={row.is_correct}
                              onChange={(e) => updateFbRow(row.lane, "is_correct", e.target.checked)}
                            />
                          </td>
                          <td>
                            <div className="coord-pair">
                              <span className="coord-lbl">x</span>
                              <input
                                type="number"
                                value={row.correct_tip_x}
                                disabled={row.is_correct}
                                onChange={(e) => updateFbRow(row.lane, "correct_tip_x", e.target.value)}
                                placeholder="—"
                              />
                              <span className="coord-lbl">y</span>
                              <input
                                type="number"
                                value={row.correct_tip_y}
                                disabled={row.is_correct}
                                onChange={(e) => updateFbRow(row.lane, "correct_tip_y", e.target.value)}
                                placeholder="—"
                              />
                            </div>
                          </td>
                          <td>
                            <textarea
                              rows={1}
                              value={row.comment}
                              onChange={(e) => updateFbRow(row.lane, "comment", e.target.value)}
                              placeholder="optional…"
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <div className="submit-row">
                  <span className="fb-note" id="fb-note" />
                  <button className="submit-btn" onClick={submitFeedback} disabled={fbLoading}>
                    {fbLoading ? <><span className="spinner" />Submitting…</> : "✓ Submit feedback"}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
