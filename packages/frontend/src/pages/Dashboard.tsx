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
  fetchBinanceMarkPrice,
  Candle,
  RealtimeKlineMessage,
  connectRealtimeKline,
  fetchChart,
  fetchHealth,
  fetchPrediction,
  fetchSymbols,
  getAuthToken,
  Timeframe,
} from "@/services/api";
import {
  PREDICTION_PIPELINE_STEPS,
  PredictionStageDefinition,
  PredictionStageKey,
  PredictionStageProgress,
} from "@/types/predictionProgress";

const HISTORICAL_LIMIT_BY_TIMEFRAME: Record<Timeframe, number> = {
  "1m": 900,
  "5m": 950,
  "15m": 1050,
  "1h": 1150,
  "4h": 1250,
  "1d": 1000,
  "1w": 800,
};
const MIN_CANDLE_TARGET_BY_TIMEFRAME: Partial<Record<Timeframe, number>> = {
  "4h": 320,
  "1d": 300,
  "1w": 220,
};
const INITIAL_FAST_LOAD_LIMIT = 700;
const HOT_PRELOAD_LIMIT = 600;
const BACKGROUND_HISTORY_TARGET_LIMIT = 4200;
const BACKGROUND_HISTORY_MIN_TARGET = 3200;
const CHART_CACHE_TTL_MS = 5 * 60 * 1000;
const CHART_SWITCH_DEBOUNCE_MS = 180;
const HOT_PRELOAD_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XAUTUSDT"] as const;
const MAX_CANDLE_BUFFER = 12000;
const BINANCE_TICKER_REFRESH_MS = 45_000;
const CHART_FETCH_TIMEOUT_MS = 18_000;
const CHART_SWITCH_RECOVERY_DELAY_MS = 1200;
const PREDICTION_STAGE_MIN_VISIBLE_MS = 180;
const PREDICTION_RENDER_SETTLE_MS = 120;

const FORECAST_HORIZON_OPTIONS = [
  { label: "6H", hours: 6 },
  { label: "24H", hours: 24 },
  { label: "3D", hours: 72 },
  { label: "7D", hours: 168 },
];

const SUPPORTED_TIMEFRAMES: Timeframe[] = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"];

interface RegionalClockContext {
  timeZone: string;
  locationLabel: string;
}

function RegionalClockBadge({
  timeZone,
  locationLabel,
}: {
  timeZone: string;
  locationLabel: string;
}) {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNowMs(Date.now());
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  const formattedNow = useMemo(
    () =>
      new Intl.DateTimeFormat("vi-VN", {
        timeZone,
        day: "2-digit",
        month: "2-digit",
        year: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }).format(new Date(nowMs)),
    [nowMs, timeZone],
  );

  return (
    <p className="text-[11px] text-cyan-200/85">
      {formattedNow} · {locationLabel}
    </p>
  );
}

function isSupportedTimeframe(value: unknown): value is Timeframe {
  return typeof value === "string" && SUPPORTED_TIMEFRAMES.includes(value as Timeframe);
}

function timeframeChartLimit(timeframe: Timeframe): number {
  return HISTORICAL_LIMIT_BY_TIMEFRAME[timeframe] ?? 1500;
}

function timeframeOlderPageLimit(timeframe: Timeframe): number {
  return Math.max(280, Math.floor(timeframeChartLimit(timeframe) * 0.45));
}

function buildChartCacheKey(symbol: string, timeframe: Timeframe): string {
  return `${symbol.toUpperCase()}::${timeframe}`;
}

function minimumChartCandles(timeframe: Timeframe): number {
  return MIN_CANDLE_TARGET_BY_TIMEFRAME[timeframe] ?? Math.min(220, timeframeChartLimit(timeframe));
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

function waitFor(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
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
    if (error.code === "ERR_NETWORK" || error.code === "ERR_CONNECTION_RESET") {
      return "Network connection was interrupted while loading chart data. Retrying...";
    }

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

function isTransientChartError(error: unknown): boolean {
  if (!axios.isAxiosError(error)) {
    return false;
  }

  if (error.code === "ERR_CANCELED") {
    return false;
  }

  const status = error.response?.status;
  return (
    error.code === "ECONNABORTED" ||
    error.code === "ERR_NETWORK" ||
    error.code === "ERR_CONNECTION_RESET" ||
    status === 408 ||
    status === 425 ||
    status === 429 ||
    status === 500 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    typeof status !== "number"
  );
}

function isCanceledChartRequest(error: unknown): boolean {
  return axios.isAxiosError(error) && error.code === "ERR_CANCELED";
}

export default function Dashboard() {
  const navigate = useNavigate();
  const browserTimeZone =
    Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";

  const {
    token,
    symbol,
    timeframe,
    symbols,
    chartCandles,
    chartCache,
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
    setChartCacheEntry,
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
  const [binanceMarkPrice, setBinanceMarkPrice] = useState<number | null>(null);
  const [regionalClock, setRegionalClock] = useState<RegionalClockContext>({
    timeZone: browserTimeZone,
    locationLabel: `${browserTimeZone} (browser)` ,
  });
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const hasMoreHistoryRef = useRef(true);
  const lastLazyLoadAnchorRef = useRef<string | null>(null);
  const previousHorizonHoursRef = useRef<number>(selectedHorizonHours);
  const chartCandlesRef = useRef<Candle[]>(chartCandles);
  const chartCacheRef = useRef(chartCache);
  const wsWasOfflineRef = useRef(false);
  const backfillInFlightRef = useRef(false);
  const chartSwitchTimerRef = useRef<number | null>(null);
  const chartRecoveryTimerRef = useRef<number | null>(null);
  const chartLoadSequenceRef = useRef(0);
  const activeChartKeyRef = useRef<string>(buildChartCacheKey(symbol, timeframe));
  const activeChartRequestControllerRef = useRef<AbortController | null>(null);
  const activeHydrationKeysRef = useRef<Set<string>>(new Set());
  const preloadInFlightRef = useRef(false);
  const predictionStageChangedAtRef = useRef<number>(0);
  const [backgroundHydrating, setBackgroundHydrating] = useState(false);
  const [lastInitialLoadMs, setLastInitialLoadMs] = useState<number | null>(null);
  const [predictionStage, setPredictionStage] = useState<PredictionStageKey | null>(null);

  const debouncedQuery = useDebouncedValue(searchQuery, 180);

  const selectedHorizonBars = useMemo(() => {
    const bars = Math.ceil((selectedHorizonHours * 3600) / timeframeSeconds(timeframe));
    return Math.max(1, Math.min(336, bars));
  }, [selectedHorizonHours, timeframe]);

  const activePredictionStage = useMemo<PredictionStageDefinition | null>(() => {
    if (!predictionStage) {
      return null;
    }
    return PREDICTION_PIPELINE_STEPS.find((step) => step.key === predictionStage) ?? null;
  }, [predictionStage]);

  const predictionProgress = useMemo<PredictionStageProgress[]>(() => {
    if (!loadingPrediction) {
      return [];
    }

    const activeIndex = predictionStage
      ? PREDICTION_PIPELINE_STEPS.findIndex((step) => step.key === predictionStage)
      : -1;

    return PREDICTION_PIPELINE_STEPS.map((step, index) => ({
      ...step,
      status: index < activeIndex ? "done" : index === activeIndex ? "active" : "pending",
    }));
  }, [loadingPrediction, predictionStage]);

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

  const fetchChartWithBackfill = useCallback(
    async (
      targetSymbol: string,
      targetTimeframe: Timeframe,
      requestedLimit: number,
      minimumTarget: number,
      seedCandles: Candle[] = [],
    ): Promise<Candle[]> => {
      let merged = seedCandles;
      if (merged.length === 0) {
        merged = await fetchChart(targetSymbol, targetTimeframe, requestedLimit);
      }

      if (merged.length === 0) {
        return merged;
      }

      const desiredMinimum = Math.max(1, Math.min(requestedLimit, minimumTarget));
      if (merged.length >= desiredMinimum) {
        return merged;
      }

      let anchor = merged[0]?.timestamp;
      let attempts = 0;
      const maxAttempts = 10;

      while (merged.length < desiredMinimum && anchor && attempts < maxAttempts) {
        const olderBatch = await fetchChart(
          targetSymbol,
          targetTimeframe,
          timeframeOlderPageLimit(targetTimeframe),
          anchor,
        );
        if (olderBatch.length === 0) {
          break;
        }

        const nextMerged = mergeCandles(merged, olderBatch);
        if (nextMerged.length === merged.length) {
          break;
        }

        merged = nextMerged;
        const nextAnchor = merged[0]?.timestamp;
        if (!nextAnchor || nextAnchor === anchor) {
          break;
        }

        anchor = nextAnchor;
        attempts += 1;
      }

      return merged;
    },
    [],
  );

  const hydrateChartHistory = useCallback(
    async (
      targetSymbol: string,
      targetTimeframe: Timeframe,
      cacheKey: string,
      seedCandles: Candle[],
    ) => {
      if (seedCandles.length === 0 || seedCandles.length >= BACKGROUND_HISTORY_TARGET_LIMIT) {
        return;
      }
      if (activeHydrationKeysRef.current.has(cacheKey)) {
        return;
      }

      activeHydrationKeysRef.current.add(cacheKey);
      const activeAtStart = activeChartKeyRef.current === cacheKey;
      if (activeAtStart) {
        setBackgroundHydrating(true);
      }

      try {
        const hydratedCandles = await fetchChartWithBackfill(
          targetSymbol,
          targetTimeframe,
          BACKGROUND_HISTORY_TARGET_LIMIT,
          BACKGROUND_HISTORY_MIN_TARGET,
          seedCandles,
        );

        if (hydratedCandles.length > 0) {
          setChartCacheEntry(cacheKey, {
            candles: hydratedCandles,
            fetchedAt: Date.now(),
          });

          if (activeChartKeyRef.current === cacheKey) {
            setChartCandles((previous) => mergeCandles(previous, hydratedCandles));
          }
        }
      } catch (error) {
        console.warn("[Dashboard] Background chart hydration failed", error);
      } finally {
        activeHydrationKeysRef.current.delete(cacheKey);
        if (activeChartKeyRef.current === cacheKey || activeAtStart) {
          setBackgroundHydrating(false);
        }
      }
    },
    [fetchChartWithBackfill, setChartCacheEntry, setChartCandles],
  );

  useEffect(() => {
    chartCandlesRef.current = chartCandles;
  }, [chartCandles]);

  useEffect(() => {
    chartCacheRef.current = chartCache;
  }, [chartCache]);

  useEffect(() => {
    activeChartKeyRef.current = buildChartCacheKey(symbol, timeframe);
  }, [symbol, timeframe]);

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

  const loadChartProgressive = useCallback(
    async (
      targetSymbol: string,
      targetTimeframe: Timeframe,
      options: { forceRefresh?: boolean; recoveryAttempt?: number } = {},
    ) => {
      if (!token || !targetSymbol) {
        return;
      }

      const forceRefresh = options.forceRefresh ?? false;
      const recoveryAttempt = options.recoveryAttempt ?? 0;
      const cacheKey = buildChartCacheKey(targetSymbol, targetTimeframe);
      activeChartKeyRef.current = cacheKey;

      if (activeChartRequestControllerRef.current) {
        activeChartRequestControllerRef.current.abort();
      }
      const requestController = new AbortController();
      activeChartRequestControllerRef.current = requestController;

      const now = Date.now();
      const cachedEntry = chartCacheRef.current[cacheKey];
      const isFreshCache =
        !!cachedEntry && cachedEntry.candles.length > 0 && now - cachedEntry.fetchedAt <= CHART_CACHE_TTL_MS;

      if (cachedEntry && cachedEntry.candles.length > 0) {
        setChartCandles(cachedEntry.candles);
        setApiStatus("online");
        setLoadingChart(false);
      }

      if (isFreshCache && !forceRefresh) {
        setLastInitialLoadMs(0);
        return;
      }

      const sequence = ++chartLoadSequenceRef.current;
      const hasCachedChart = !!cachedEntry && cachedEntry.candles.length > 0;
      if (!hasCachedChart) {
        setChartCandles([]);
        setLoadingChart(true);
      } else {
        setLoadingChart(false);
      }
      setErrorMessage("");

      const startedAt = performance.now();
      try {
        const fastCandles = await fetchChart(
          targetSymbol,
          targetTimeframe,
          INITIAL_FAST_LOAD_LIMIT,
          undefined,
          CHART_FETCH_TIMEOUT_MS,
          {
            signal: requestController.signal,
            retries: 2,
          },
        );
        if (sequence !== chartLoadSequenceRef.current || activeChartKeyRef.current !== cacheKey) {
          return;
        }

        const initialSnapshot =
          fastCandles.length > 0
            ? mergeCandles(cachedEntry?.candles ?? [], fastCandles)
            : (cachedEntry?.candles ?? []);

        setChartCandles(initialSnapshot);
        setChartCacheEntry(cacheKey, {
          candles: initialSnapshot,
          fetchedAt: Date.now(),
        });
        setApiStatus("online");
        setLoadingChart(false);
        setLastInitialLoadMs(performance.now() - startedAt);

        if (initialSnapshot.length > 0 && initialSnapshot.length < BACKGROUND_HISTORY_TARGET_LIMIT) {
          void hydrateChartHistory(targetSymbol, targetTimeframe, cacheKey, initialSnapshot);
        }
      } catch (error) {
        if (isCanceledChartRequest(error)) {
          return;
        }

        if (sequence !== chartLoadSequenceRef.current || activeChartKeyRef.current !== cacheKey) {
          return;
        }

        setLoadingChart(false);
        if (!hasCachedChart) {
          const transientError = isTransientChartError(error);
          if (transientError && recoveryAttempt < 1) {
            setApiStatus("online");
            setErrorMessage("Network unstable while switching timeframe. Retrying chart load...");

            if (chartRecoveryTimerRef.current !== null) {
              window.clearTimeout(chartRecoveryTimerRef.current);
            }

            chartRecoveryTimerRef.current = window.setTimeout(() => {
              void loadChartProgressive(targetSymbol, targetTimeframe, {
                forceRefresh: true,
                recoveryAttempt: recoveryAttempt + 1,
              });
            }, CHART_SWITCH_RECOVERY_DELAY_MS);
            return;
          }

          setApiStatus(transientError ? "online" : "offline");
          setErrorMessage(
            resolveApiErrorMessage(
              error,
              "Unable to load chart data. Verify backend, token, and S3 parquet data.",
            ),
          );
          setChartCandles([]);
          return;
        }

        setApiStatus("online");
        setErrorMessage("Loaded cached chart data. Live refresh will retry automatically.");
      } finally {
        if (activeChartRequestControllerRef.current === requestController) {
          activeChartRequestControllerRef.current = null;
        }
      }
    },
    [
      token,
      hydrateChartHistory,
      setLoadingChart,
      setErrorMessage,
      setChartCandles,
      setApiStatus,
      setChartCacheEntry,
    ],
  );

  const backfillLatestCandles = useCallback(async () => {
    if (!token || !symbol) {
      return;
    }

    try {
      const latestSnapshotLimit = Math.max(360, Math.min(1800, timeframeChartLimit(timeframe)));
      const candles = await fetchChartWithBackfill(
        symbol,
        timeframe,
        latestSnapshotLimit,
        Math.max(900, minimumChartCandles(timeframe)),
      );
      if (candles.length === 0) {
        return;
      }

      const cacheKey = buildChartCacheKey(symbol, timeframe);
      setChartCandles((previous) => mergeCandles(previous, candles));
      setChartCacheEntry(cacheKey, {
        candles: mergeCandles(chartCandlesRef.current, candles),
        fetchedAt: Date.now(),
      });
      setApiStatus("online");
    } catch (error) {
      if (!isTransientChartError(error)) {
        setApiStatus("offline");
      }
    }
  }, [token, symbol, timeframe, fetchChartWithBackfill, setChartCandles, setChartCacheEntry, setApiStatus]);

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
        const cacheKey = buildChartCacheKey(symbol, timeframe);
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
        setChartCacheEntry(cacheKey, {
          candles: mergeCandles(chartCandlesRef.current, olderCandles),
          fetchedAt: Date.now(),
        });

        setApiStatus("online");
      } catch (error) {
        lastLazyLoadAnchorRef.current = null;
        setApiStatus(isTransientChartError(error) ? "online" : "offline");
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
      setChartCacheEntry,
      setErrorMessage,
    ],
  );

  const advancePredictionStage = useCallback(async (nextStage: PredictionStageKey) => {
    const previousMark = predictionStageChangedAtRef.current;
    if (previousMark > 0) {
      const elapsed = performance.now() - previousMark;
      if (elapsed < PREDICTION_STAGE_MIN_VISIBLE_MS) {
        await waitFor(PREDICTION_STAGE_MIN_VISIBLE_MS - elapsed);
      }
    }

    predictionStageChangedAtRef.current = performance.now();
    setPredictionStage(nextStage);
  }, []);

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
    predictionStageChangedAtRef.current = 0;

    try {
      await advancePredictionStage("sync_market_data");
      const freshSnapshotLimit = Math.max(240, Math.min(1200, timeframeChartLimit(timeframe)));
      const freshestChartCandles = await fetchChartWithBackfill(
        symbol,
        timeframe,
        freshSnapshotLimit,
        Math.max(900, minimumChartCandles(timeframe)),
      );

      await advancePredictionStage("fetch_external_context");
      const predictionSource = freshestChartCandles.length > 0 ? freshestChartCandles : chartCandles;

      if (predictionSource.length < 20) {
        throw new Error("Prediction requires at least 20 candles in chart data.");
      }

      if (freshestChartCandles.length > 0) {
        setChartCandles((previous) => mergeCandles(previous, freshestChartCandles));
        setChartCacheEntry(buildChartCacheKey(symbol, timeframe), {
          candles: mergeCandles(chartCandlesRef.current, freshestChartCandles),
          fetchedAt: Date.now(),
        });
      }

      await advancePredictionStage("prepare_payload");
      const latestCandles = predictionSource.slice(-500);

      await advancePredictionStage("send_request");
      const predictionPromise = fetchPrediction({
        symbol,
        timeframe,
        latest_candles: latestCandles,
        horizon: selectedHorizonBars,
      });

      await advancePredictionStage("run_inference");
      const result = await predictionPromise;

      await advancePredictionStage("render_output");

      const anchorCandle = predictionSource[predictionSource.length - 1];
      const anchor = {
        baseTimestamp: anchorCandle.timestamp,
        baseClose: anchorCandle.close,
      };

      setPrediction(result);
      setPredictionAnchor(anchor);
      setApiStatus("online");
      await waitFor(PREDICTION_RENDER_SETTLE_MS);
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
      setPredictionStage(null);
      predictionStageChangedAtRef.current = 0;
    }
  }, [
    advancePredictionStage,
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
    fetchChartWithBackfill,
    setChartCacheEntry,
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
    let cancelled = false;
    const controller = new AbortController();

    const resolveRegionalClock = async () => {
      try {
        const response = await fetch("https://worldtimeapi.org/api/ip", {
          signal: controller.signal,
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(`worldtimeapi unavailable: ${response.status}`);
        }

        const payload = (await response.json()) as {
          timezone?: string;
          city?: string;
          region?: string;
          country_name?: string;
        };

        if (cancelled) {
          return;
        }

        const resolvedTimeZone =
          typeof payload.timezone === "string" && payload.timezone
            ? payload.timezone
            : browserTimeZone;

        const locationParts = [payload.city, payload.region, payload.country_name]
          .filter((part): part is string => typeof part === "string" && part.length > 0)
          .slice(0, 2);

        const resolvedLocationLabel =
          locationParts.length > 0
            ? `${locationParts.join(", ")} · ${resolvedTimeZone}`
            : `${resolvedTimeZone} (IP region)`;

        setRegionalClock({
          timeZone: resolvedTimeZone,
          locationLabel: resolvedLocationLabel,
        });
      } catch {
        if (cancelled) {
          return;
        }

        setRegionalClock({
          timeZone: browserTimeZone,
          locationLabel: `${browserTimeZone} (browser)`,
        });
      }
    };

    void resolveRegionalClock();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [browserTimeZone]);

  useEffect(() => {
    void loadSymbols();
  }, [loadSymbols]);

  useEffect(() => {
    if (!token || !symbol) {
      return;
    }

    hasMoreHistoryRef.current = true;
    lastLazyLoadAnchorRef.current = null;
    wsWasOfflineRef.current = false;
    backfillInFlightRef.current = false;
    setLoadingOlderCandles(false);

    if (activeChartRequestControllerRef.current) {
      activeChartRequestControllerRef.current.abort();
      activeChartRequestControllerRef.current = null;
    }

    if (chartRecoveryTimerRef.current !== null) {
      window.clearTimeout(chartRecoveryTimerRef.current);
      chartRecoveryTimerRef.current = null;
    }

    const cacheKey = buildChartCacheKey(symbol, timeframe);
    const now = Date.now();
    const cachedEntry = chartCacheRef.current[cacheKey];
    const hasCachedChart = !!cachedEntry && cachedEntry.candles.length > 0;
    const isFreshCache = hasCachedChart && now - cachedEntry.fetchedAt <= CHART_CACHE_TTL_MS;

    if (hasCachedChart) {
      setChartCandles(cachedEntry.candles);
      setApiStatus("online");
      setLoadingChart(false);
    }

    if (chartSwitchTimerRef.current !== null) {
      window.clearTimeout(chartSwitchTimerRef.current);
      chartSwitchTimerRef.current = null;
    }

    if (isFreshCache) {
      setLastInitialLoadMs(0);
      return;
    }

    if (!hasCachedChart) {
      setChartCandles([]);
      setLoadingChart(true);
    }

    chartSwitchTimerRef.current = window.setTimeout(() => {
      void loadChartProgressive(symbol, timeframe);
    }, CHART_SWITCH_DEBOUNCE_MS);

    return () => {
      if (chartSwitchTimerRef.current !== null) {
        window.clearTimeout(chartSwitchTimerRef.current);
        chartSwitchTimerRef.current = null;
      }

      if (chartRecoveryTimerRef.current !== null) {
        window.clearTimeout(chartRecoveryTimerRef.current);
        chartRecoveryTimerRef.current = null;
      }
    };
  }, [
    token,
    symbol,
    timeframe,
    hydrateChartHistory,
    loadChartProgressive,
    setApiStatus,
    setChartCandles,
    setLoadingChart,
  ]);

  useEffect(() => {
    if (!token || loadingChart) {
      return;
    }
    if (preloadInFlightRef.current) {
      return;
    }

    let cancelled = false;
    let preloadTimer: number | null = null;
    preloadInFlightRef.current = true;

    const preloadHotCharts = async () => {
      try {
        for (const hotSymbol of HOT_PRELOAD_SYMBOLS) {
          if (cancelled) {
            break;
          }

          const cacheKey = buildChartCacheKey(hotSymbol, timeframe);
          const cachedEntry = chartCacheRef.current[cacheKey];
          if (cachedEntry && Date.now() - cachedEntry.fetchedAt <= CHART_CACHE_TTL_MS) {
            continue;
          }

          try {
            const candles = await fetchChart(hotSymbol, timeframe, HOT_PRELOAD_LIMIT, undefined, 10_000, {
              retries: 1,
            });
            if (cancelled || candles.length === 0) {
              continue;
            }

            setChartCacheEntry(cacheKey, {
              candles,
              fetchedAt: Date.now(),
            });
          } catch {
            // Silent preload failure should not block dashboard UX.
          }
        }
      } finally {
        preloadInFlightRef.current = false;
      }
    };

    preloadTimer = window.setTimeout(() => {
      void preloadHotCharts();
    }, 1200);

    return () => {
      cancelled = true;
      if (preloadTimer !== null) {
        window.clearTimeout(preloadTimer);
      }
      preloadInFlightRef.current = false;
    };
  }, [token, timeframe, loadingChart, setChartCacheEntry]);

  useEffect(() => {
    return () => {
      if (activeChartRequestControllerRef.current) {
        activeChartRequestControllerRef.current.abort();
        activeChartRequestControllerRef.current = null;
      }
      if (chartRecoveryTimerRef.current !== null) {
        window.clearTimeout(chartRecoveryTimerRef.current);
        chartRecoveryTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!token || !symbol) {
      setBinanceTicker(null);
      setBinanceMarkPrice(null);
      return;
    }

    let cancelled = false;
    const refreshBinanceTicker = async () => {
      const [tickerResult, markPriceResult] = await Promise.allSettled([
        fetchBinance24hTicker(symbol),
        fetchBinanceMarkPrice(symbol),
      ]);

      if (cancelled) {
        return;
      }

      if (tickerResult.status === "fulfilled") {
        setBinanceTicker({
          lastPrice: tickerResult.value.lastPrice,
          changePercent: tickerResult.value.changePercent,
        });
      } else {
        setBinanceTicker(null);
      }

      if (markPriceResult.status === "fulfilled") {
        setBinanceMarkPrice(markPriceResult.value.markPrice);
      } else {
        setBinanceMarkPrice(null);
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
      void loadChartProgressive(symbol, timeframe, { forceRefresh: true });
    }, 45000);

    return () => window.clearInterval(interval);
  }, [token, wsStatus, symbol, timeframe, loadChartProgressive]);

  useEffect(() => {
    if (!token) {
      wsWasOfflineRef.current = false;
      backfillInFlightRef.current = false;
      return;
    }

    if (wsStatus === "offline") {
      wsWasOfflineRef.current = true;
      return;
    }

    if (wsStatus === "online" && wsWasOfflineRef.current && !backfillInFlightRef.current) {
      wsWasOfflineRef.current = false;
      backfillInFlightRef.current = true;
      void backfillLatestCandles().finally(() => {
        backfillInFlightRef.current = false;
      });
    }
  }, [token, wsStatus, backfillLatestCandles]);

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

      const expectedCandleIntervalMs = timeframeSeconds(timeframe) * 1000;
      const realtimeGapThresholdMs = expectedCandleIntervalMs * 1.6;

      const socket = connectRealtimeKline(
        symbol,
        timeframe,
        (message) => {
          if (message.symbol.toUpperCase() !== symbol.toUpperCase() || message.timeframe !== timeframe) {
            return;
          }

          retryCount = 0;
          const existingCandles = chartCandlesRef.current;
          const previousLastMs =
            existingCandles.length > 0
              ? toEpochMs(existingCandles[existingCandles.length - 1].timestamp)
              : Number.NaN;
          const incomingMs = toEpochMs(message.timestamp);
          const hasRealtimeGap =
            Number.isFinite(previousLastMs) &&
            Number.isFinite(incomingMs) &&
            incomingMs > previousLastMs + realtimeGapThresholdMs;

          if (hasRealtimeGap && !backfillInFlightRef.current) {
            backfillInFlightRef.current = true;
            void backfillLatestCandles().finally(() => {
              backfillInFlightRef.current = false;
            });
          }

          setChartCandles((previous) => {
            const next = upsertRealtimeCandle(previous, message);
            const cacheKey = buildChartCacheKey(symbol, timeframe);
            setChartCacheEntry(cacheKey, {
              candles: next,
              fetchedAt: Date.now(),
            });
            return next;
          });
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
  }, [token, symbol, timeframe, backfillLatestCandles, setChartCandles, setChartCacheEntry, setWsStatus]);

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
              <div className="text-right">
                <p className="text-xs text-violet-200/70">
                  {loadingChart
                    ? "Loading recent candles..."
                    : backgroundHydrating
                      ? `Hydrating deeper history... (${chartCandles.length})`
                    : loadingOlderCandles
                      ? `Loading older candles... (${chartCandles.length})`
                      : `Candles loaded: ${chartCandles.length}${
                          lastInitialLoadMs !== null && lastInitialLoadMs > 0
                            ? ` · initial ${Math.round(lastInitialLoadMs)}ms`
                            : ""
                        }`}
                </p>
                <RegionalClockBadge
                  timeZone={regionalClock.timeZone}
                  locationLabel={regionalClock.locationLabel}
                />
              </div>
            </div>

            <TradingChart
              key={`${symbol}-${timeframe}`}
              symbol={symbol}
              candles={chartCandles}
              timeframe={timeframe}
              prediction={prediction}
              predictionAnchor={predictionAnchor}
              timeZone={regionalClock.timeZone}
              onRequestOlderCandles={loadOlderCandles}
              isLoadingOlder={loadingOlderCandles}
              isSyncing={loadingChart}
              isPredicting={loadingPrediction}
              predictionProgress={predictionProgress}
              activePredictionStage={activePredictionStage}
            />
          </section>

          <PredictionPanel
            symbol={symbol}
            timeframe={timeframe}
            lastPrice={binanceMarkPrice ?? ticker.price}
            currentPriceLabel={binanceMarkPrice !== null ? "Current (Mark Price)" : "Current (Last Price)"}
            lastCandleTimestamp={chartCandles[chartCandles.length - 1]?.timestamp ?? null}
            prediction={prediction}
            horizonOptions={FORECAST_HORIZON_OPTIONS}
            selectedHorizonHours={selectedHorizonHours}
            selectedHorizonBars={selectedHorizonBars}
            loading={loadingPrediction}
            predictionProgress={predictionProgress}
            activePredictionStage={activePredictionStage}
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
