import { useCallback, useState } from "react";

import { AiCouncilDecision, fetchAiCouncil, Timeframe, TradeAction } from "@/services/api";

interface AiCouncilPanelProps {
  symbol: string;
  timeframe: Timeframe;
}

const ACTION_STYLES: Record<TradeAction, { badge: string; label: string }> = {
  LONG: { badge: "bg-emerald-500/20 text-emerald-300 border-emerald-400/50", label: "LONG" },
  SHORT: { badge: "bg-red-500/20 text-red-300 border-red-400/50", label: "SHORT" },
  HOLD: { badge: "bg-zinc-500/20 text-zinc-300 border-zinc-400/50", label: "HOLD" },
};

export default function AiCouncilPanel({ symbol, timeframe }: AiCouncilPanelProps) {
  const [loading, setLoading] = useState(false);
  const [decision, setDecision] = useState<AiCouncilDecision | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAsk = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDecision(null);

    try {
      const result = await fetchAiCouncil(symbol, timeframe);
      setDecision(result);
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : "AI Council is unavailable. Check GEMINI_API_KEY configuration.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  const style = decision ? ACTION_STYLES[decision.action] : null;

  return (
    <div className="mt-4 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-4">
      <p className="muted-label">AI Council</p>

      {/* Ask button */}
      <button
        type="button"
        onClick={handleAsk}
        disabled={loading}
        className="mt-2 w-full rounded-xl border border-violet-300/60 bg-gradient-to-r from-violet-500/15 via-fuchsia-500/10 to-cyan-500/10 px-4 py-3 text-sm font-semibold text-violet-100 transition hover:from-violet-500/25 hover:via-fuchsia-500/20 hover:to-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-violet-300 border-t-transparent" />
            Agents are analyzing the market...
          </span>
        ) : (
          "🔮 Ask AI Council"
        )}
      </button>

      {/* Error */}
      {error && (
        <p className="mt-3 rounded-lg border border-red-400/30 bg-red-500/10 p-2.5 text-xs text-red-200">
          {error}
        </p>
      )}

      {/* Result */}
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
        </div>
      )}
    </div>
  );
}
