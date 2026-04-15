import { Timeframe } from "@/services/api";
import logoEye from "@/assets/logo-eye.svg";

interface TopBarProps {
  symbolSearch: string;
  onSymbolSearchChange: (value: string) => void;
  symbol: string;
  livePrice: number | null;
  changePct: number | null;
  changeLabel: string;
  apiStatus: "idle" | "online" | "offline";
  wsStatus: "idle" | "connecting" | "online" | "offline";
  timeframe: Timeframe;
  onTimeframeChange: (value: Timeframe) => void;
  isAuthenticated: boolean;
  onOpenAuthModal: () => void;
  onSignOut: () => void;
}

const timeframes: Timeframe[] = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"];

export default function TopBar({
  symbolSearch,
  onSymbolSearchChange,
  symbol,
  livePrice,
  changePct,
  changeLabel,
  apiStatus,
  wsStatus,
  timeframe,
  onTimeframeChange,
  isAuthenticated,
  onOpenAuthModal,
  onSignOut,
}: TopBarProps) {
  const isPositive = (changePct ?? 0) >= 0;

  return (
    <header className="glass-panel rounded-2xl px-4 py-3">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex min-w-56 items-center gap-3">
          <img src={logoEye} alt="AetherForecast" className="h-10 w-16" />
          <div>
            <p className="text-sm font-semibold tracking-wide text-neon-cyan">AetherForecast</p>
            <p className="text-[11px] uppercase tracking-[0.12em] text-violet-200/70">
              Cosmic Trading Dashboard
            </p>
          </div>
        </div>

        <label className="min-w-56 flex-1">
          <span className="muted-label">Search Symbol</span>
          <input
            className="mt-1 w-full rounded-lg border border-violet-400/35 bg-cosmic-900/70 px-3 py-2 text-sm outline-none ring-neon-cyan/50 transition focus:ring"
            value={symbolSearch}
            onChange={(event) => onSymbolSearchChange(event.target.value.toUpperCase())}
            placeholder="BTCUSDT"
          />
        </label>

        <div className="flex min-w-56 items-center gap-2 rounded-lg border border-violet-400/30 bg-cosmic-900/70 px-3 py-2">
          <div>
            <p className="muted-label">Ticker</p>
            <p className="font-medium text-violet-100">{symbol}</p>
          </div>
          <div className="ml-auto text-right">
            <p className="font-mono text-sm text-violet-100">
              {livePrice !== null ? livePrice.toFixed(6) : "--"}
            </p>
            <p className={`text-xs ${isPositive ? "price-up" : "price-down"}`}>
              {changePct !== null ? `${isPositive ? "+" : ""}${changePct.toFixed(2)}%` : "--"}
            </p>
            <p className="text-[10px] uppercase tracking-[0.12em] text-violet-300/70">{changeLabel}</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {timeframes.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => onTimeframeChange(item)}
              className={`rounded-lg border px-3 py-2 text-xs font-medium transition ${
                timeframe === item
                  ? "border-cyan-300/80 bg-cyan-400/15 text-cyan-100"
                  : "border-violet-400/35 bg-cosmic-900/70 text-violet-200 hover:border-violet-300/60"
              }`}
            >
              {item}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span
            className={`rounded-full border px-3 py-1 text-xs ${
              apiStatus === "online"
                ? "border-cyan-400/80 text-cyan-200"
                : apiStatus === "offline"
                  ? "border-rose-400/70 text-rose-200"
                  : "border-violet-300/40 text-violet-200"
            }`}
          >
            API {apiStatus}
          </span>
          <span
            className={`rounded-full border px-3 py-1 text-xs ${
              wsStatus === "online"
                ? "border-cyan-400/80 text-cyan-200"
                : wsStatus === "connecting"
                  ? "border-amber-400/70 text-amber-200"
                  : wsStatus === "offline"
                    ? "border-rose-400/70 text-rose-200"
                    : "border-violet-300/40 text-violet-200"
            }`}
          >
            WS {wsStatus}
          </span>
          <button
            type="button"
            onClick={onOpenAuthModal}
            className="rounded-lg border border-violet-400/35 bg-violet-500/10 px-3 py-2 text-xs text-violet-100 transition hover:bg-violet-500/20"
          >
            {isAuthenticated ? "Account" : "Login / Register"}
          </button>
          {isAuthenticated && (
            <button
              type="button"
              onClick={onSignOut}
              className="rounded-lg border border-violet-400/35 px-3 py-2 text-xs text-violet-200 transition hover:border-violet-300/60"
            >
              Sign out
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
