import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

import TradingChart from "@/components/chart/TradingChart";
import PredictionPanel from "@/components/layout/PredictionPanel";
import Sidebar from "@/components/layout/Sidebar";
import TopBar from "@/components/layout/TopBar";
import AuthTokenModal from "@/components/ui/AuthTokenModal";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useMarketStore } from "@/hooks/useMarketStore";
import {
  fetchBinance24hTicker,
  Candle,
  PredictResponse,
  RealtimeKlineMessage,
  connectRealtimeKline,
  fetchChart,
  fetchHealth,
  fetchPrediction,
  fetchSymbols,
  getAuthToken,
  Timeframe,
} from "@/services/api";

const HISTORICAL_LIMIT_BY_TIMEFRAME: Record<Timeframe, number> = {
  "1m": 900,
  "5m": 950,
  "15m": 1050,
  "1h": 1150,
  "4h": 1250,
  "1d": 1000,
  "1w": 800,
};
const MAX_CANDLE_BUFFER = 12000;
const BINANCE_TICKER_REFRESH_MS = 45_000;

const FORECAST_HORIZON_OPTIONS = [
  { label: "6H", hours: 6 },
  { label: "24H", hours: 24 },
  { label: "3D", hours: 72 },
  { label: "7D", hours: 168 },
];

const SUPPORTED_TIMEFRAMES: Timeframe[] = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"];

function isSupportedTimeframe(value: unknown): value is Timeframe {
  return typeof value === "string" && SUPPORTED_TIMEFRAMES.includes(value as Timeframe);
}

function timeframeChartLimit(timeframe: Timeframe): number {
  return HISTORICAL_LIMIT_BY_TIMEFRAME[timeframe] ?? 1500;
}

function timeframeOlderPageLimit(timeframe: Timeframe): number {
  return Math.max(280, Math.floor(timeframeChartLimit(timeframe) * 0.45));
}

function toEpochMs(timestamp: string): number {
  return new Date(timestamp).getTime();
}

function timeframeSeconds(timeframe: Timeframe): number {
  switch (timeframe) {
    case "1m":
      return 60;
    case "5m":
      return 5 * 60;
    case "15m":
      return 15 * 60;
    case "1h":
      return 60 * 60;
    case "4h":
      return 4 * 60 * 60;
    case "1d":
      return 24 * 60 * 60;
    case "1w":
      return 7 * 24 * 60 * 60;
    default:
      return 60 * 60;
  }
}

function computeTickerChange(candles: Candle[]): { changePct: number | null; label: string } {
  if (candles.length < 2) {
    return { changePct: null, label: "24h" };
  }

  const lastCandle = candles[candles.length - 1];
  const lastPrice = Number(lastCandle.close);
  const lastTimestampMs = toEpochMs(lastCandle.timestamp);
  if (!Number.isFinite(lastPrice) || !Number.isFinite(lastTimestampMs) || lastPrice <= 0) {
    return { changePct: null, label: "24h" };
  }

  const targetMs = lastTimestampMs - 24 * 60 * 60 * 1000;
  let referenceCandle = candles[0];
  for (let index = candles.length - 1; index >= 0; index -= 1) {
    const candidateTs = toEpochMs(candles[index].timestamp);
    if (!Number.isFinite(candidateTs)) {
      continue;
    }
    if (candidateTs <= targetMs) {
      referenceCandle = candles[index];
      break;
    }
  }

  const referencePrice = Number(referenceCandle.close);
  if (!Number.isFinite(referencePrice) || referencePrice <= 0) {
    return { changePct: null, label: "24h" };
  }

  const usedWindowFallback = toEpochMs(candles[0].timestamp) > targetMs;
  return {
    changePct: ((lastPrice - referencePrice) / referencePrice) * 100,
    label: usedWindowFallback ? "window" : "24h",
  };
}

function mergeCandles(previous: Candle[], incoming: Candle[], maxCandles = MAX_CANDLE_BUFFER): Candle[] {
  if (incoming.length === 0) {
    return previous;
  }

  const mergedByTimestamp = new Map<number, Candle>();
  for (const candle of previous) {
    const timestampMs = toEpochMs(candle.timestamp);
    if (Number.isFinite(timestampMs)) {
      mergedByTimestamp.set(timestampMs, candle);
    }
  }

  for (const candle of incoming) {
    const timestampMs = toEpochMs(candle.timestamp);
    if (Number.isFinite(timestampMs)) {
      mergedByTimestamp.set(timestampMs, candle);
    }
  }

  const merged = [...mergedByTimestamp.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([, candle]) => candle);

  if (merged.length <= maxCandles) {
    return merged;
  }
  return merged.slice(-maxCandles);
}

function upsertRealtimeCandle(
  previous: Candle[],
  incoming: RealtimeKlineMessage,
  maxCandles = MAX_CANDLE_BUFFER,
): Candle[] {
  const nextCandle: Candle = {
    timestamp: incoming.timestamp,
    open: incoming.open,
    high: incoming.high,
    low: incoming.low,
    close: incoming.close,
    volume: incoming.volume,
  };

  if (previous.length === 0) {
    return [nextCandle];
  }

  const next = [...previous];
  const incomingMs = toEpochMs(nextCandle.timestamp);
  const lastIndex = next.length - 1;
  const lastMs = toEpochMs(next[lastIndex].timestamp);

  if (incomingMs === lastMs) {
    next[lastIndex] = nextCandle;
    return next;
  }

  if (incomingMs > lastMs) {
    next.push(nextCandle);
    return next.slice(-maxCandles);
  }

  const existingIndex = next.findIndex((item) => toEpochMs(item.timestamp) === incomingMs);
  if (existingIndex >= 0) {
    next[existingIndex] = nextCandle;
    return next;
  }

  next.push(nextCandle);
  next.sort((left, right) => toEpochMs(left.timestamp) - toEpochMs(right.timestamp));
  return next.slice(-maxCandles);
}

function resolveApiErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    if (status === 401 || status === 403) {
      return "Authentication token is invalid or expired. Please sign in again.";
    }
    if (status === 404) {
      return "Requested API route was not found. Please verify backend deployment.";
    }
    if (status && status >= 500) {
      return "Backend is temporarily unavailable. Please retry in a moment.";
    }
  }
  return fallback;
}

export default function Dashboard() {
  const navigate = useNavigate();

  const {
    token,
    symbol,
    timeframe,
    symbols,
    chartCandles,
    prediction,
    loadingSymbols,
    loadingChart,
    loadingPrediction,
    apiStatus,
    wsStatus,
    errorMessage,
    setToken,
    clearToken,
    setSymbol,
    setTimeframe,
    setSymbols,
    setChartCandles,
    setPrediction,
    setLoadingSymbols,
    setLoadingChart,
    setLoadingPrediction,
    setApiStatus,
    setWsStatus,
    setErrorMessage,
  } = useMarketStore();

  const [searchQuery, setSearchQuery] = useState("");
  const [showAuthModal, setShowAuthModal] = useState(!token);
  const [selectedHorizonHours, setSelectedHorizonHours] = useState<number>(24);
  const [loadingOlderCandles, setLoadingOlderCandles] = useState(false);
  const [predictionAnchor, setPredictionAnchor] = useState<{
    baseTimestamp: string;
    baseClose: number;
  } | null>(null);
  const [binanceTicker, setBinanceTicker] = useState<{
    lastPrice: number;
    changePercent: number;
  } | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const hasMoreHistoryRef = useRef(true);
  const lastLazyLoadAnchorRef = useRef<string | null>(null);
  const previousHorizonHoursRef = useRef<number>(selectedHorizonHours);

  const debouncedQuery = useDebouncedValue(searchQuery, 180);

  const selectedHorizonBars = useMemo(() => {
    const bars = Math.ceil((selectedHorizonHours * 3600) / timeframeSeconds(timeframe));
    return Math.max(1, Math.min(336, bars));
  }, [selectedHorizonHours, timeframe]);

  const filteredSymbols = useMemo(() => {
    if (!debouncedQuery) {
      return symbols;
    }
    return symbols.filter((item) => item.includes(debouncedQuery));
  }, [symbols, debouncedQuery]);

  const ticker = useMemo(() => {
    if (chartCandles.length === 0) {
      return {
        price: null as number | null,
        changePct: null as number | null,
        changeLabel: "24h",
      };
    }

    const last = chartCandles[chartCandles.length - 1]?.close ?? null;
    const { changePct: fallbackChangePct, label: fallbackLabel } = computeTickerChange(chartCandles);
    const usingBinanceTicker =
      binanceTicker !== null && Number.isFinite(binanceTicker.lastPrice) && Number.isFinite(binanceTicker.changePercent);

    return {
      price: usingBinanceTicker ? binanceTicker.lastPrice : last,
      changePct: usingBinanceTicker ? binanceTicker.changePercent : fallbackChangePct,
      changeLabel: usingBinanceTicker ? "binance24h" : fallbackLabel,
    };
  }, [chartCandles, binanceTicker]);

  const loadSymbols = useCallback(async () => {
    if (!token) {
      return;
    }

    setLoadingSymbols(true);
    setErrorMessage("");
    try {
      const payload = await fetchSymbols();
      const symbolList = payload.length > 0 ? payload : ["BTCUSDT", "ETHUSDT"];
      setSymbols(symbolList);
      if (!symbolList.includes(symbol)) {
        setSymbol(symbolList[0]);
      }
    } catch (error) {
      setErrorMessage(
        resolveApiErrorMessage(
          error,
          "Unable to load symbols. Check JWT token and backend availability.",
        ),
      );
    } finally {
      setLoadingSymbols(false);
    }
  }, [token, symbol, setLoadingSymbols, setErrorMessage, setSymbols, setSymbol]);

  const loadChart = useCallback(async () => {
    if (!token || !symbol) {
      return;
    }

    setLoadingChart(true);
    setErrorMessage("");

    try {
      const candles = await fetchChart(symbol, timeframe, timeframeChartLimit(timeframe));
      setChartCandles(candles);
      setApiStatus("online");
    } catch (error) {
      setApiStatus("offline");
      setErrorMessage(
        resolveApiErrorMessage(
          error,
          "Unable to load chart data. Verify backend, token, and S3 parquet data.",
        ),
      );
      setChartCandles([]);
    } finally {
      setLoadingChart(false);
    }
  }, [token, symbol, timeframe, setLoadingChart, setErrorMessage, setChartCandles, setApiStatus]);

  const loadOlderCandles = useCallback(
    async (oldestTimestamp: string) => {
      if (!token || !symbol) {
        return;
      }
      if (loadingChart || loadingOlderCandles || !hasMoreHistoryRef.current) {
        return;
      }
      if (!oldestTimestamp || lastLazyLoadAnchorRef.current === oldestTimestamp) {
        return;
      }

      lastLazyLoadAnchorRef.current = oldestTimestamp;
      setLoadingOlderCandles(true);

      try {
        const olderCandles = await fetchChart(
          symbol,
          timeframe,
          timeframeOlderPageLimit(timeframe),
          oldestTimestamp,
        );

        if (olderCandles.length === 0) {
          hasMoreHistoryRef.current = false;
          return;
        }

        setChartCandles((previous) => {
          const previousOldestMs = previous.length > 0 ? toEpochMs(previous[0].timestamp) : Number.POSITIVE_INFINITY;
          const merged = mergeCandles(previous, olderCandles);
          const mergedOldestMs = merged.length > 0 ? toEpochMs(merged[0].timestamp) : Number.POSITIVE_INFINITY;

          if (mergedOldestMs >= previousOldestMs) {
            hasMoreHistoryRef.current = false;
          }

          return merged;
        });

        setApiStatus("online");
      } catch (error) {
        lastLazyLoadAnchorRef.current = null;
        setApiStatus("offline");
        setErrorMessage(
          resolveApiErrorMessage(
            error,
            "Unable to load older candles. Please retry after a moment.",
          ),
        );
      } finally {
        setLoadingOlderCandles(false);
      }
    },
    [
      token,
      symbol,
      timeframe,
      loadingChart,
      loadingOlderCandles,
      setApiStatus,
      setChartCandles,
      setErrorMessage,
    ],
  );

  const runPrediction = useCallback(async () => {
    if (!token) {
      setShowAuthModal(true);
      return;
    }

    if (chartCandles.length < 20) {
      setErrorMessage("Prediction requires at least 20 candles in chart data.");
      return;
    }

    setLoadingPrediction(true);
    setErrorMessage("");

    try {
      const freshSnapshotLimit = Math.max(240, Math.min(1200, timeframeChartLimit(timeframe)));
      const freshestChartCandles = await fetchChart(symbol, timeframe, freshSnapshotLimit);
      const predictionSource = freshestChartCandles.length > 0 ? freshestChartCandles : chartCandles;

      if (predictionSource.length < 20) {
        throw new Error("Prediction requires at least 20 candles in chart data.");
      }

      if (freshestChartCandles.length > 0) {
        setChartCandles((previous) => mergeCandles(previous, freshestChartCandles));
      }

      const latestCandles = predictionSource.slice(-500);
      const result = await fetchPrediction({
        symbol,
        timeframe,
        latest_candles: latestCandles,
        horizon: selectedHorizonBars,
      });

      const anchorCandle = predictionSource[predictionSource.length - 1];
      const anchor = {
        baseTimestamp: anchorCandle.timestamp,
        baseClose: anchorCandle.close,
      };

      setPrediction(result);
      setPredictionAnchor(anchor);
      setApiStatus("online");
    } catch (error) {
      setApiStatus("offline");
      setErrorMessage(
        resolveApiErrorMessage(
          error,
          "Prediction request failed. Verify token and backend /predict endpoint.",
        ),
      );
    } finally {
      setLoadingPrediction(false);
    }
  }, [
    token,
    chartCandles,
    symbol,
    timeframe,
    selectedHorizonBars,
    setChartCandles,
    setLoadingPrediction,
    setErrorMessage,
    setPrediction,
    setApiStatus,
  ]);

  useEffect(() => {
    const tokenFromStorage = getAuthToken();
    if (tokenFromStorage && !token) {
      setToken(tokenFromStorage);
      setShowAuthModal(false);
    }
  }, [token, setToken]);

  useEffect(() => {
    const checkApi = async () => {
      try {
        await fetchHealth();
        setApiStatus("online");
      } catch {
        setApiStatus("offline");
      }
    };

    void checkApi();
  }, [setApiStatus]);

  useEffect(() => {
    void loadSymbols();
  }, [loadSymbols]);

  useEffect(() => {
    void loadChart();
  }, [loadChart]);

  useEffect(() => {
    hasMoreHistoryRef.current = true;
    lastLazyLoadAnchorRef.current = null;
    setLoadingOlderCandles(false);
    setChartCandles([]);
  }, [symbol, timeframe, setChartCandles]);

  useEffect(() => {
    if (!token || !symbol) {
      setBinanceTicker(null);
      return;
    }

    let cancelled = false;
    const refreshBinanceTicker = async () => {
      try {
        const nextTicker = await fetchBinance24hTicker(symbol);
        if (cancelled) {
          return;
        }
        setBinanceTicker({
          lastPrice: nextTicker.lastPrice,
          changePercent: nextTicker.changePercent,
        });
      } catch {
        if (!cancelled) {
          setBinanceTicker(null);
        }
      }
    };

    void refreshBinanceTicker();
    const interval = window.setInterval(() => {
      void refreshBinanceTicker();
    }, BINANCE_TICKER_REFRESH_MS);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [token, symbol]);

  useEffect(() => {
    setPrediction(null);
    setPredictionAnchor(null);
  }, [symbol, timeframe, setPrediction]);

  useEffect(() => {
    if (previousHorizonHoursRef.current === selectedHorizonHours) {
      return;
    }

    previousHorizonHoursRef.current = selectedHorizonHours;
    setPrediction(null);
    setPredictionAnchor(null);
  }, [selectedHorizonHours, setPrediction]);

  useEffect(() => {
    if (!token || wsStatus === "online") {
      return;
    }

    const interval = window.setInterval(() => {
      void loadChart();
    }, 45000);

    return () => window.clearInterval(interval);
  }, [token, wsStatus, loadChart]);

  useEffect(() => {
    if (!token) {
      setWsStatus("idle");
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
      return;
    }

    let closedByEffect = false;
    let retryCount = 0;

    const connect = () => {
      if (closedByEffect) {
        return;
      }

      const socket = connectRealtimeKline(
        symbol,
        timeframe,
        (message) => {
          if (message.symbol.toUpperCase() !== symbol.toUpperCase() || message.timeframe !== timeframe) {
            return;
          }

          retryCount = 0;
          setChartCandles((previous) => upsertRealtimeCandle(previous, message));
        },
        (status) => {
          setWsStatus(status);
        },
      );

      socketRef.current = socket;

      socket.onclose = () => {
        setWsStatus("offline");
        if (closedByEffect) {
          return;
        }

        const retryDelayMs = Math.min(10000, 1500 * (retryCount + 1));
        retryCount += 1;
        reconnectTimerRef.current = window.setTimeout(connect, retryDelayMs);
      };

      socket.onerror = () => {
        setWsStatus("offline");
      };
    };

    connect();

    return () => {
      closedByEffect = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }

      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, [token, symbol, timeframe, setChartCandles, setWsStatus]);

  const handleAuthenticated = (value: string) => {
    setToken(value);
    setShowAuthModal(false);
    setErrorMessage("");
  };

  const handleSignOut = () => {
    clearToken();
    setShowAuthModal(false);
    setSymbols([]);
    setChartCandles([]);
    setPrediction(null);
    setPredictionAnchor(null);
    navigate("/", { replace: true });
  };

  return (
    <div className="cosmic-shell p-3 lg:p-4">
      <div className="mx-auto flex max-w-[1880px] flex-col gap-4">
        <TopBar
          symbolSearch={searchQuery}
          onSymbolSearchChange={setSearchQuery}
          symbol={symbol}
          livePrice={ticker.price}
          changePct={ticker.changePct}
          changeLabel={ticker.changeLabel}
          apiStatus={apiStatus}
          wsStatus={wsStatus}
          timeframe={timeframe}
          onTimeframeChange={(value: Timeframe) => setTimeframe(value)}
          isAuthenticated={Boolean(token)}
          onOpenAuthModal={() => setShowAuthModal(true)}
          onSignOut={handleSignOut}
        />

        {errorMessage && (
          <div className="rounded-xl border border-rose-400/50 bg-rose-500/15 px-4 py-2 text-sm text-rose-100">
            {errorMessage}
          </div>
        )}

        <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)_360px]">
          <Sidebar
            symbols={filteredSymbols}
            selectedSymbol={symbol}
            searchQuery={searchQuery}
            loading={loadingSymbols}
            onSearchQueryChange={setSearchQuery}
            onSelectSymbol={setSymbol}
          />

          <section className="glass-panel rounded-2xl p-3 lg:p-4">
            <div className="mb-3 flex items-center justify-between gap-4 rounded-lg border border-violet-400/20 bg-cosmic-900/45 px-3 py-2">
              <div>
                <p className="muted-label">Chart</p>
                <h2 className="text-lg font-semibold text-violet-50">
                  {symbol} Candles ({timeframe})
                </h2>
              </div>
              <p className="text-xs text-violet-200/70">
                {loadingChart
                  ? "Syncing chart..."
                  : loadingOlderCandles
                    ? `Loading older candles... (${chartCandles.length})`
                    : `Candles loaded: ${chartCandles.length}`}
              </p>
            </div>

            <TradingChart
              key={`${symbol}-${timeframe}`}
              symbol={symbol}
              candles={chartCandles}
              timeframe={timeframe}
              prediction={prediction}
              predictionAnchor={predictionAnchor}
              onRequestOlderCandles={loadOlderCandles}
              isLoadingOlder={loadingOlderCandles}
              isSyncing={loadingChart || wsStatus === "connecting"}
              isPredicting={loadingPrediction}
            />
          </section>

          <PredictionPanel
            symbol={symbol}
            timeframe={timeframe}
            lastPrice={ticker.price}
            lastCandleTimestamp={chartCandles[chartCandles.length - 1]?.timestamp ?? null}
            prediction={prediction}
            horizonOptions={FORECAST_HORIZON_OPTIONS}
            selectedHorizonHours={selectedHorizonHours}
            selectedHorizonBars={selectedHorizonBars}
            loading={loadingPrediction}
            onSelectHorizon={setSelectedHorizonHours}
            onPredict={runPrediction}
          />
        </div>
      </div>

      <AuthTokenModal
        open={showAuthModal}
        hasToken={Boolean(token)}
        onAuthenticate={handleAuthenticated}
        onSignOut={handleSignOut}
        onClose={() => setShowAuthModal(false)}
      />
    </div>
  );
}
