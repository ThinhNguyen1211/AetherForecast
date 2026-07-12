import { useCallback, useEffect, useRef, useState } from "react";

import { getAuthToken, Timeframe, TradeAction, AiCouncilDecision } from "@/services/api";
import logoEye from "@/assets/logo-eye.svg";

interface AiCouncilPanelProps {
  symbol: string;
  timeframe: Timeframe;
  hasPrediction: boolean;
}

const ACTION_STYLES: Record<TradeAction, { badge: string; glow: string; label: string }> = {
  LONG: {
    badge: "bg-emerald-500/20 text-emerald-300 border-emerald-400/50",
    glow: "shadow-[0_0_15px_rgba(16,185,129,0.25)]",
    label: "🟢 LONG",
  },
  SHORT: {
    badge: "bg-red-500/20 text-red-300 border-red-400/50",
    glow: "shadow-[0_0_15px_rgba(239,68,68,0.25)]",
    label: "🔴 SHORT",
  },
  HOLD: {
    badge: "bg-zinc-500/20 text-zinc-300 border-zinc-400/50",
    glow: "shadow-[0_0_15px_rgba(161,161,170,0.15)]",
    label: "⏸ HOLD",
  },
};

// Agent label styling for terminal output
const AGENT_COLORS: Record<string, string> = {
  "📊 Quant Analyst": "text-cyan-400",
  "🛡️ Risk Manager": "text-amber-400",
  "⚖️ Execution Judge": "text-fuchsia-400",
};

function getAgentColor(line: string): string {
  for (const [label, color] of Object.entries(AGENT_COLORS)) {
    if (line.includes(label)) return color;
  }
  return "text-green-300/90";
}

function TerminalLine({ line, index }: { line: string; index: number }) {
  const isAgentHeader = Object.keys(AGENT_COLORS).some((label) => line.includes(label));
  const isSystem = line.startsWith("🚀") || line.startsWith("📊") || line.startsWith("✅");
  const isThought = line.startsWith("💭");

  return (
    <div
      className={`whitespace-pre-wrap break-words ${isAgentHeader ? "mt-2 pt-2 border-t border-violet-400/10" : ""}`}
    >
      <span className="text-violet-400/40 select-none mr-1 text-[10px]">
        {String(index + 1).padStart(2, "0")}
      </span>
      {isAgentHeader ? (
        <span className={`font-semibold ${getAgentColor(line)}`}>{line}</span>
      ) : isSystem ? (
        <span className="text-violet-300/80">{line}</span>
      ) : isThought ? (
        <span className="text-green-300/60 italic">{line}</span>
      ) : (
        <span className="text-green-300/90">{line}</span>
      )}
    </div>
  );
}

export default function AiCouncilPanel({ symbol, timeframe, hasPrediction }: AiCouncilPanelProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiLogs, setAiLogs] = useState<string[]>([]);
  const [finalDecision, setFinalDecision] = useState<AiCouncilDecision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTranscript, setShowTranscript] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll terminal to bottom
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [aiLogs]);

  const handleAnalyze = useCallback(async () => {
    if (isAnalyzing) return;

    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    setIsAnalyzing(true);
    setError(null);
    setFinalDecision(null);
    setAiLogs(["Establishing secure connection to Aether AI Council..."]);
    setShowTranscript(false);

    const token = getAuthToken();
    if (!token) {
      setError("Authentication required. Please sign in.");
      setIsAnalyzing(false);
      return;
    }

    try {
      const response = await fetch("/api/ai/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ symbol, timeframe }),
        signal: controller.signal,
      });

      if (response.status === 429) {
        setError("⏳ Rate limit reached. Please try again later (max 5 requests/hour).");
        setIsAnalyzing(false);
        return;
      }

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(errorBody || `HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error("Streaming not supported by this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const event of events) {
          const lines = event
            .split("\n")
            .filter((l) => l.startsWith("data: "))
            .map((l) => l.slice(6));

          const message = lines.join("\n").trim();
          if (!message || message === "[KEEPALIVE]" || message === "[CONNECTED]") continue;

          // Final result — extract strictly the JSON object after [FINAL_RESULT],
          // immune to trailing newlines, whitespace, or SSE fragment leftovers.
          const finalMatch = message.match(/\[FINAL_RESULT\]:?\s*(\{.*\})/s);
          if (finalMatch && finalMatch[1]) {
            try {
              const cleanJson = finalMatch[1].trim();
              const parsed = JSON.parse(cleanJson) as AiCouncilDecision;
              setFinalDecision(parsed);
              setIsAnalyzing(false);
              setAiLogs((prev) => [
                ...prev,
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "✅ AI Agent Team has reached consensus.",
                `   Action: ${parsed.action} | Confidence: ${(parsed.confidence * 100).toFixed(0)}%`,
              ]);
            } catch (parseErr) {
              console.error("Parse Error Final JSON:", parseErr, message);
              setAiLogs((prev) => [...prev, `⚠️ Failed to parse result — raw output saved to console.`]);
            }
            continue;
          }

          // Error event
          if (message.startsWith("[ERROR]:")) {
            const errMsg = message.slice("[ERROR]:".length);
            setError(errMsg);
            setAiLogs((prev) => [...prev, `❌ ${errMsg}`]);
            continue;
          }

          // Traceback event — display full stack trace in the terminal
          if (message.startsWith("[TRACE]:")) {
            const trace = message.slice("[TRACE]:".length);
            setAiLogs((prev) => [...prev, `🔍 Traceback: ${trace}`]);
            continue;
          }

          // Regular stream line
          setAiLogs((prev) => [...prev, message]);
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      const message =
        err instanceof Error
          ? err.message
          : "AI Agent Team is unavailable. Check DEEPSEEK_API_KEY configuration.";
      setError(message);
    } finally {
      setIsAnalyzing(false);
    }
  }, [symbol, timeframe, isAnalyzing]);

  const style = finalDecision ? ACTION_STYLES[finalDecision.action] : null;

  return (
    <div className="mt-4 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-4">
      <p className="muted-label">AI AGENT TEAM</p>

      {/* ── Main Action Button ── */}
      <button
        type="button"
        onClick={handleAnalyze}
        disabled={isAnalyzing || !hasPrediction}
        title={
          !hasPrediction
            ? "Run ML Prediction first to provide data for the AI Agent Team."
            : undefined
        }
        className="mt-2 flex w-full items-center justify-center gap-3 rounded-xl border border-violet-300/60 bg-gradient-to-r from-violet-500/15 via-fuchsia-500/10 to-cyan-500/10 px-4 py-3.5 text-sm font-semibold text-violet-100 transition-all duration-300 hover:from-violet-500/25 hover:via-fuchsia-500/20 hover:to-cyan-500/20 hover:shadow-[0_0_20px_rgba(139,92,246,0.15)] disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isAnalyzing ? (
          <>
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-violet-300 border-t-transparent flex-shrink-0" />
            <span>Agents are analyzing strategy...</span>
          </>
        ) : (
          <>
            <img
              src={logoEye}
              alt="Aether AI"
              className="w-10 h-10 flex-shrink-0 object-contain opacity-90 drop-shadow-[0_0_6px_rgba(139,92,246,0.5)]"
            />
            <span>Analyze &amp; Generate Signals</span>
          </>
        )}
      </button>

      {!hasPrediction && !isAnalyzing && (
        <p className="mt-1.5 text-center text-[10px] text-violet-200/50">
          Run ML Prediction first to unlock this feature.
        </p>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="mt-3 rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-xs text-red-200">
          <span className="font-semibold">Error: </span>
          {error}
        </div>
      )}

      {/* ── Live Terminal ── */}
      {isAnalyzing && aiLogs.length > 0 && (
        <div className="mt-3 rounded-xl border border-violet-400/20 bg-[#0c0c14] overflow-hidden">
          {/* Terminal header bar */}
          <div className="flex items-center gap-2 border-b border-violet-400/10 bg-[#12121f] px-3 py-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-green-500/70" />
            <span className="ml-2 text-[10px] text-violet-200/50 font-mono">
              aether-council — {symbol} @ {timeframe}
            </span>
            <span className="ml-auto inline-block h-2 w-2 animate-pulse rounded-full bg-green-400/80" />
          </div>
          {/* Terminal body */}
          <div
            ref={terminalRef}
            className="max-h-72 overflow-y-auto p-3 font-mono text-[11px] leading-relaxed scrollbar-slim"
          >
            {aiLogs.map((line, i) => (
              <TerminalLine key={i} line={line} index={i} />
            ))}
            <span className="inline-block animate-pulse text-green-400">▊</span>
          </div>
        </div>
      )}

      {/* ── Decision Result Card ── */}
      {finalDecision && style && !isAnalyzing && (
        <div className="mt-3 space-y-3">
          {/* Action + Confidence badge row */}
          <div className={`flex items-center gap-3 rounded-xl border p-3 ${style.badge} ${style.glow}`}>
            <span className="text-xl font-black tracking-wider">{style.label}</span>
            <div className="ml-auto flex flex-col items-end">
              <span className="text-[10px] uppercase tracking-wider opacity-60">Confidence</span>
              <span className="text-lg font-bold">{(finalDecision.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Entry / SL / TP / Leverage grid */}
          {finalDecision.action !== "HOLD" && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-lg border border-violet-400/20 bg-cosmic-900/50 p-2.5">
                <p className="text-[10px] font-medium uppercase tracking-wider text-violet-200/50">Entry Price</p>
                <p className="mt-1 text-base font-bold text-cyan-200">
                  ${finalDecision.entry.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                </p>
              </div>
              <div className="rounded-lg border border-violet-400/20 bg-cosmic-900/50 p-2.5">
                <p className="text-[10px] font-medium uppercase tracking-wider text-violet-200/50">Leverage</p>
                <p className="mt-1 text-base font-bold text-cyan-200">{finalDecision.leverage}×</p>
              </div>
              <div className="rounded-lg border border-red-400/15 bg-red-500/5 p-2.5">
                <p className="text-[10px] font-medium uppercase tracking-wider text-red-300/50">Stop Loss</p>
                <p className="mt-1 text-base font-bold text-red-300">
                  ${finalDecision.stop_loss.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                </p>
              </div>
              <div className="rounded-lg border border-emerald-400/15 bg-emerald-500/5 p-2.5">
                <p className="text-[10px] font-medium uppercase tracking-wider text-emerald-300/50">Take Profit</p>
                <p className="mt-1 text-base font-bold text-emerald-300">
                  ${finalDecision.take_profit.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                </p>
              </div>
            </div>
          )}

          {/* Risk/Reward Ratio */}
          {finalDecision.action !== "HOLD" && finalDecision.stop_loss > 0 && finalDecision.take_profit > 0 && (
            <div className="rounded-lg border border-violet-400/15 bg-cosmic-900/40 p-2.5">
              <p className="text-[10px] font-medium uppercase tracking-wider text-violet-200/50">Risk / Reward Ratio</p>
              <p className="mt-0.5 text-sm font-bold text-violet-200">
                1 :{" "}
                {Math.abs(
                  (finalDecision.take_profit - finalDecision.entry) /
                  (finalDecision.entry - finalDecision.stop_loss || 1)
                ).toFixed(2)}
              </p>
            </div>
          )}

          {/* Council Reasoning */}
          <div className="rounded-xl border border-violet-400/20 bg-cosmic-900/50 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-violet-200/50 mb-2">
              Council Reasoning
            </p>
            <p className="text-xs leading-relaxed text-violet-100/90">
              {finalDecision.reasoning}
            </p>
          </div>

          {/* Transcript toggle */}
          {aiLogs.length > 0 && (
            <>
              <button
                type="button"
                onClick={() => setShowTranscript((prev) => !prev)}
                className="flex items-center gap-1.5 text-[10px] text-violet-200/50 hover:text-violet-200/80 transition-colors"
              >
                <span className="inline-block transition-transform duration-200" style={{ transform: showTranscript ? "rotate(180deg)" : "rotate(0deg)" }}>
                  ▼
                </span>
                {showTranscript ? "Hide agent transcript" : "View full agent transcript"}
              </button>

              {showTranscript && (
                <div className="rounded-xl border border-violet-400/15 bg-[#0c0c14] overflow-hidden">
                  <div className="flex items-center gap-2 border-b border-violet-400/10 bg-[#12121f] px-3 py-1.5">
                    <span className="h-2 w-2 rounded-full bg-violet-400/50" />
                    <span className="text-[10px] text-violet-200/40 font-mono">transcript — {symbol}</span>
                  </div>
                  <div className="max-h-56 overflow-y-auto p-3 font-mono text-[11px] leading-relaxed scrollbar-slim">
                    {aiLogs.map((line, i) => (
                      <TerminalLine key={i} line={line} index={i} />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
