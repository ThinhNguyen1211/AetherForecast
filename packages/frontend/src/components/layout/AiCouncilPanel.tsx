import { useCallback, useEffect, useRef, useState } from "react";

import { getAuthToken, Timeframe, TradeAction, AiCouncilDecision } from "@/services/api";
import logoEye from "@/assets/logo-eye.svg";

interface AiCouncilPanelProps {
  symbol: string;
  timeframe: Timeframe;
  hasPrediction: boolean;
}

const ACTION_STYLES: Record<TradeAction, { badge: string; label: string }> = {
  LONG: { badge: "bg-emerald-500/20 text-emerald-300 border-emerald-400/50", label: "LONG" },
  SHORT: { badge: "bg-red-500/20 text-red-300 border-red-400/50", label: "SHORT" },
  HOLD: { badge: "bg-zinc-500/20 text-zinc-300 border-zinc-400/50", label: "HOLD" },
};

function resolveApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) {
    return configured;
  }
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      return "http://localhost:8000";
    }
    return `${window.location.protocol}//${window.location.hostname}`;
  }
  return "http://localhost:8000";
}

export default function AiCouncilPanel({ symbol, timeframe, hasPrediction }: AiCouncilPanelProps) {
  const [loading, setLoading] = useState(false);
  const [decision, setDecision] = useState<AiCouncilDecision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streamLines, setStreamLines] = useState<string[]>([]);
  const [showTerminal, setShowTerminal] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll terminal to bottom when new lines arrive
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [streamLines]);

  const handleAsk = useCallback(async () => {
    if (loading) return;

    // Abort any in-flight request
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setDecision(null);
    setStreamLines([]);
    setShowTerminal(true);

    const token = getAuthToken();
    if (!token) {
      setError("Authentication required. Please sign in.");
      setLoading(false);
      return;
    }

    try {
      const baseUrl = resolveApiBaseUrl();
      const response = await fetch(`${baseUrl}/api/ai/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ symbol: symbol, timeframe: timeframe }),
        signal: controller.signal,
      });

      // Handle rate limit
      if (response.status === 429) {
        setError("⏳ Hội đồng đang nghỉ ngơi, vui lòng thử lại sau (giới hạn 5 lần/giờ).");
        setLoading(false);
        setShowTerminal(false);
        return;
      }

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(errorBody || `HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No response body (streaming not supported)");
      }

      // Read SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        // Keep incomplete last chunk in buffer
        buffer = events.pop() || "";

        for (const event of events) {
          const lines = event
            .split("\n")
            .filter((l) => l.startsWith("data: "))
            .map((l) => l.slice(6));

          const message = lines.join("\n").trim();
          if (!message || message === "[KEEPALIVE]") continue;

          // Check for final result
          if (message.startsWith("[FINAL_RESULT]:")) {
            const jsonStr = message.slice("[FINAL_RESULT]:".length);
            try {
              const parsed = JSON.parse(jsonStr) as AiCouncilDecision;
              setDecision(parsed);
              setStreamLines((prev) => [
                ...prev,
                "",
                "✅ Hội đồng AI đã đưa ra quyết định.",
              ]);
            } catch (parseErr) {
              setStreamLines((prev) => [
                ...prev,
                `⚠️ Lỗi parse kết quả: ${parseErr}`,
              ]);
            }
            continue;
          }

          // Check for error
          if (message.startsWith("[ERROR]:")) {
            const errMsg = message.slice("[ERROR]:".length);
            setError(errMsg);
            continue;
          }

          // Regular stream line — typewriter append
          setStreamLines((prev) => [...prev, message]);
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      const message =
        err instanceof Error
          ? err.message
          : "AI Council is unavailable. Check GEMINI_API_KEY configuration.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe, loading]);

  const style = decision ? ACTION_STYLES[decision.action] : null;

  return (
    <div className="mt-4 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-4">
      <p className="muted-label">AI Council</p>

      {/* Ask button */}
      <button
        type="button"
        onClick={handleAsk}
        disabled={loading || !hasPrediction}
        title={
          !hasPrediction
            ? "Hãy chạy ML Prediction trước để cung cấp dữ liệu cho Hội đồng AI."
            : undefined
        }
        className="mt-2 w-full rounded-xl border border-violet-300/60 bg-gradient-to-r from-violet-500/15 via-fuchsia-500/10 to-cyan-500/10 px-4 py-3 text-sm font-semibold text-violet-100 transition hover:from-violet-500/25 hover:via-fuchsia-500/20 hover:to-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-violet-300 border-t-transparent" />
            Agents đang họp chiến lược...
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            <img src={logoEye} alt="Aether AI" className="h-4 w-4 opacity-90" />
            Aether AI Agents: Analyze & Generate Signals
          </span>
        )}
      </button>
      {!hasPrediction && !loading && (
        <p className="mt-1.5 text-center text-[10px] text-violet-200/50">
          Chạy ML Prediction trước để mở khóa tính năng này.
        </p>
      )}

      {/* Error */}
      {error && (
        <p className="mt-3 rounded-lg border border-red-400/30 bg-red-500/10 p-2.5 text-xs text-red-200">
          {error}
        </p>
      )}

      {/* Terminal Console */}
      {showTerminal && streamLines.length > 0 && !decision && (
        <div
          ref={terminalRef}
          className="mt-3 max-h-64 overflow-y-auto rounded-lg border border-violet-400/20 bg-black/90 p-3 font-mono text-[11px] leading-relaxed text-green-300/90 scrollbar-slim"
        >
          {streamLines.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-words">
              <span className="text-violet-400/60 select-none">{">"} </span>
              {line}
            </div>
          ))}
          {loading && (
            <span className="inline-block animate-pulse text-green-400">▊</span>
          )}
        </div>
      )}

      {/* Result Card */}
      {decision && style && (
        <div className="mt-3 space-y-3">
          {/* Action badge */}
          <div className="flex items-center gap-3">
            <span
              className={`rounded-lg border px-3 py-1.5 text-sm font-bold ${style.badge}`}
            >
              {style.label}
            </span>
            <span className="text-xs text-violet-200/70">
              Confidence: {(decision.confidence * 100).toFixed(0)}%
            </span>
          </div>

          {/* Entry / SL / TP / Leverage grid */}
          {decision.action !== "HOLD" && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-lg border border-violet-400/20 bg-cosmic-900/50 p-2">
                <p className="text-violet-200/60">Entry</p>
                <p className="mt-0.5 font-semibold text-cyan-200">
                  {decision.entry.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                </p>
              </div>
              <div className="rounded-lg border border-violet-400/20 bg-cosmic-900/50 p-2">
                <p className="text-violet-200/60">Leverage</p>
                <p className="mt-0.5 font-semibold text-cyan-200">{decision.leverage}x</p>
              </div>
              <div className="rounded-lg border border-violet-400/20 bg-cosmic-900/50 p-2">
                <p className="text-violet-200/60">Stop Loss</p>
                <p className="mt-0.5 font-semibold text-red-300">
                  {decision.stop_loss.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                </p>
              </div>
              <div className="rounded-lg border border-violet-400/20 bg-cosmic-900/50 p-2">
                <p className="text-violet-200/60">Take Profit</p>
                <p className="mt-0.5 font-semibold text-emerald-300">
                  {decision.take_profit.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                </p>
              </div>
            </div>
          )}

          {/* Reasoning */}
          <div className="rounded-lg border border-violet-400/20 bg-cosmic-900/50 p-2.5">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-violet-200/60">
              Council Reasoning
            </p>
            <p className="mt-1 text-xs leading-relaxed text-violet-100/90">
              {decision.reasoning}
            </p>
          </div>

          {/* Show/hide terminal transcript */}
          {streamLines.length > 0 && (
            <button
              type="button"
              onClick={() => setShowTerminal((prev) => !prev)}
              className="text-[10px] text-violet-200/50 hover:text-violet-200/80 transition"
            >
              {showTerminal ? "▲ Ẩn nhật ký họp" : "▼ Xem nhật ký họp"}
            </button>
          )}

          {showTerminal && streamLines.length > 0 && (
            <div
              ref={terminalRef}
              className="max-h-48 overflow-y-auto rounded-lg border border-violet-400/20 bg-black/90 p-3 font-mono text-[11px] leading-relaxed text-green-300/90 scrollbar-slim"
            >
              {streamLines.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-words">
                  <span className="text-violet-400/60 select-none">{">"} </span>
                  {line}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
