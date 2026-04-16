import { useEffect, useMemo, useRef } from "react";
import {
  BusinessDay,
  CandlestickData,
  ColorType,
  HistogramData,
  IChartApi,
  LineData,
  LineStyle,
  LogicalRange,
  MouseEventParams,
  TickMarkType,
  Time,
  UTCTimestamp,
  createChart,
} from "lightweight-charts";

import { Candle, PredictResponse } from "@/services/api";

interface TradingChartProps {
  symbol: string;
  candles: Candle[];
  timeframe: string;
  timeZone?: string;
  prediction: PredictResponse | null;
  predictionAnchor?: {
    baseTimestamp: string;
    baseClose: number;
  } | null;
  onRequestOlderCandles?: (oldestTimestamp: string) => void;
  isLoadingOlder?: boolean;
  isSyncing?: boolean;
  isPredicting?: boolean;
}

interface IndicatorLine {
  time: UTCTimestamp;
  value: number;
}

interface ForecastHoverPoint {
  time: UTCTimestamp;
  low: number;
  mid: number;
  high: number;
}

function parseTimestampMs(timestamp: string): number {
  const direct = new Date(timestamp).getTime();
  if (Number.isFinite(direct)) {
    return direct;
  }

  const normalized = timestamp.includes("T") ? timestamp : timestamp.replace(" ", "T");
  const withZone = /[zZ]|[+-]\d{2}:\d{2}$/.test(normalized) ? normalized : `${normalized}Z`;
  const parsed = Date.parse(withZone);
  if (Number.isFinite(parsed)) {
    return parsed;
  }

  const numeric = Number(timestamp);
  if (Number.isFinite(numeric)) {
    return numeric > 10_000_000_000 ? numeric : numeric * 1000;
  }

  return Number.NaN;
}

function toUnix(timestamp: string): UTCTimestamp {
  return Math.floor(parseTimestampMs(timestamp) / 1000) as UTCTimestamp;
}

function chartTimeToDate(time: Time): Date {
  if (typeof time === "number") {
    return new Date(Number(time) * 1000);
  }

  if (typeof time === "string") {
    const normalized = time.includes("T") ? time : `${time}T00:00:00Z`;
    const parsed = new Date(normalized);
    if (Number.isFinite(parsed.getTime())) {
      return parsed;
    }
  }

  const businessDay = time as BusinessDay;
  const candidate = new Date(Date.UTC(businessDay.year, businessDay.month - 1, businessDay.day));
  if (Number.isFinite(candidate.getTime())) {
    return candidate;
  }

  return new Date(0);
}

function chartTimeToUnix(time: Time): UTCTimestamp {
  return Math.floor(chartTimeToDate(time).getTime() / 1000) as UTCTimestamp;
}

function formatAxisTimeLabel(time: Time, timeZone: string, tickMarkType: TickMarkType): string {
  const date = chartTimeToDate(time);
  if (!Number.isFinite(date.getTime())) {
    return "";
  }

  switch (tickMarkType) {
    case TickMarkType.Year:
      return new Intl.DateTimeFormat("vi-VN", {
        timeZone,
        year: "numeric",
      }).format(date);
    case TickMarkType.Month:
      return new Intl.DateTimeFormat("vi-VN", {
        timeZone,
        month: "2-digit",
        year: "2-digit",
      }).format(date);
    case TickMarkType.DayOfMonth:
      return new Intl.DateTimeFormat("vi-VN", {
        timeZone,
        day: "2-digit",
        month: "2-digit",
      }).format(date);
    case TickMarkType.TimeWithSeconds:
      return new Intl.DateTimeFormat("vi-VN", {
        timeZone,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }).format(date);
    default:
      return new Intl.DateTimeFormat("vi-VN", {
        timeZone,
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }).format(date);
  }
}

function formatCrosshairTimeLabel(time: Time, timeZone: string): string {
  const date = chartTimeToDate(time);
  if (!Number.isFinite(date.getTime())) {
    return "";
  }

  return new Intl.DateTimeFormat("vi-VN", {
    timeZone,
    day: "2-digit",
    month: "2-digit",
    year: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function forecastRangeCapRatio(timeframe: string): number {
  switch (timeframe) {
    case "1m":
      return 0.0035;
    case "5m":
      return 0.005;
    case "15m":
      return 0.0075;
    case "1h":
      return 0.012;
    case "4h":
      return 0.02;
    case "1d":
      return 0.035;
    case "1w":
      return 0.08;
    default:
      return 0.012;
  }
}

function sanitizeCandles(candles: Candle[]): Candle[] {
  const byTime = new Map<number, Candle>();

  for (const candle of candles) {
    const epochMs = parseTimestampMs(candle.timestamp);
    if (!Number.isFinite(epochMs)) {
      continue;
    }

    const open = Number(candle.open);
    const high = Number(candle.high);
    const low = Number(candle.low);
    const close = Number(candle.close);
    const volume = Number(candle.volume);

    if (![open, high, low, close, volume].every((value) => Number.isFinite(value))) {
      continue;
    }

    const fixedHigh = Math.max(high, open, close, low);
    const fixedLow = Math.min(low, open, close, fixedHigh);

    byTime.set(epochMs, {
      timestamp: new Date(epochMs).toISOString(),
      open,
      high: fixedHigh,
      low: fixedLow,
      close,
      volume: Math.max(0, volume),
    });
  }

  return [...byTime.entries()]
    .sort((left, right) => left[0] - right[0])
    .map((item) => item[1]);
}

function timeframeSeconds(timeframe: string): number {
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

function isNearRealtime(
  range: LogicalRange,
  candleCount: number,
  toleranceBars = 2,
  futureToleranceBars = 0.2,
): boolean {
  if (candleCount <= 0) {
    return true;
  }

  const lastLogicalIndex = candleCount - 1;
  return range.to >= lastLogicalIndex - toleranceBars && range.to <= lastLogicalIndex + futureToleranceBars;
}

function calculateRsi(candles: Candle[], period = 14): IndicatorLine[] {
  if (candles.length <= period) {
    return [];
  }

  const output: IndicatorLine[] = [];
  let gains = 0;
  let losses = 0;

  for (let index = 1; index <= period; index += 1) {
    const delta = candles[index].close - candles[index - 1].close;
    if (delta >= 0) {
      gains += delta;
    } else {
      losses += Math.abs(delta);
    }
  }

  let averageGain = gains / period;
  let averageLoss = losses / period;

  for (let index = period + 1; index < candles.length; index += 1) {
    const delta = candles[index].close - candles[index - 1].close;
    const gain = delta > 0 ? delta : 0;
    const loss = delta < 0 ? Math.abs(delta) : 0;

    averageGain = (averageGain * (period - 1) + gain) / period;
    averageLoss = (averageLoss * (period - 1) + loss) / period;

    const rs = averageLoss === 0 ? 100 : averageGain / averageLoss;
    const rsi = 100 - 100 / (1 + rs);

    output.push({
      time: toUnix(candles[index].timestamp),
      value: rsi,
    });
  }

  return output;
}

function calculateEma(values: number[], period: number): Array<number | null> {
  const output: Array<number | null> = Array(values.length).fill(null);
  if (values.length < period) {
    return output;
  }

  const multiplier = 2 / (period + 1);
  let ema = values.slice(0, period).reduce((sum, value) => sum + value, 0) / period;
  output[period - 1] = ema;

  for (let index = period; index < values.length; index += 1) {
    ema = (values[index] - ema) * multiplier + ema;
    output[index] = ema;
  }

  return output;
}

function calculateMacd(candles: Candle[]): {
  signal: LineData<UTCTimestamp>[];
  histogram: HistogramData<UTCTimestamp>[];
} {
  const closes = candles.map((item) => item.close);
  const ema12 = calculateEma(closes, 12);
  const ema26 = calculateEma(closes, 26);

  const macdLine: Array<number | null> = closes.map((_, index) => {
    if (ema12[index] === null || ema26[index] === null) {
      return null;
    }
    return (ema12[index] as number) - (ema26[index] as number);
  });

  const macdValues = macdLine.map((item) => item ?? 0);
  const signalEma = calculateEma(macdValues, 9);

  const signal: LineData<UTCTimestamp>[] = [];
  const histogram: HistogramData<UTCTimestamp>[] = [];

  for (let index = 0; index < candles.length; index += 1) {
    if (macdLine[index] === null || signalEma[index] === null) {
      continue;
    }

    const time = toUnix(candles[index].timestamp);
    const signalValue = signalEma[index] as number;
    const histValue = (macdLine[index] as number) - signalValue;

    signal.push({ time, value: signalValue });
    histogram.push({
      time,
      value: histValue,
      color: histValue >= 0 ? "rgba(0,255,255,0.65)" : "rgba(255,94,140,0.7)",
    });
  }

  return { signal, histogram };
}

export default function TradingChart({
  symbol,
  candles,
  timeframe,
  timeZone,
  prediction,
  predictionAnchor,
  onRequestOlderCandles,
  isLoadingOlder = false,
  isSyncing = false,
  isPredicting = false,
}: TradingChartProps) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const mainChartRef = useRef<HTMLDivElement | null>(null);
  const rsiChartRef = useRef<HTMLDivElement | null>(null);
  const macdChartRef = useRef<HTMLDivElement | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const mainChartApiRef = useRef<IChartApi | null>(null);
  const rsiChartApiRef = useRef<IChartApi | null>(null);
  const macdChartApiRef = useRef<IChartApi | null>(null);

  const candleSeriesRef = useRef<ReturnType<IChartApi["addCandlestickSeries"]> | null>(null);
  const forecastCandleSeriesRef = useRef<ReturnType<IChartApi["addCandlestickSeries"]> | null>(null);
  const predictionLineSeriesRef = useRef<ReturnType<IChartApi["addLineSeries"]> | null>(null);
  const hoverHighGuideSeriesRef = useRef<ReturnType<IChartApi["addLineSeries"]> | null>(null);
  const hoverMidGuideSeriesRef = useRef<ReturnType<IChartApi["addLineSeries"]> | null>(null);
  const hoverLowGuideSeriesRef = useRef<ReturnType<IChartApi["addLineSeries"]> | null>(null);

  const rsiSeriesRef = useRef<ReturnType<IChartApi["addLineSeries"]> | null>(null);
  const macdHistogramRef = useRef<ReturnType<IChartApi["addHistogramSeries"]> | null>(null);
  const macdSignalRef = useRef<ReturnType<IChartApi["addLineSeries"]> | null>(null);

  const previousTimeframeRef = useRef<string>(timeframe);
  const hasInitialFitRef = useRef(false);
  const shouldAutoFollowRef = useRef(true);
  const lastRenderedLengthRef = useRef(0);
  const lastRenderedLastTimeRef = useRef<number>(Number.NaN);
  const lastRenderedFirstTimeRef = useRef<number>(Number.NaN);
  const lazyLoadAnchorRef = useRef<string | null>(null);
  const normalizedCandlesRef = useRef<Candle[]>([]);
  const loadOlderCallbackRef = useRef<((oldestTimestamp: string) => void) | null>(null);
  const loadingOlderRef = useRef(false);
  const forecastHoverPointsRef = useRef<ForecastHoverPoint[]>([]);
  const forecastStepSecondsRef = useRef<number>(timeframeSeconds(timeframe));
  const resolvedTimeZone =
    timeZone || Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  const timeZoneRef = useRef<string>(resolvedTimeZone);
  const normalizedInputCandles = useMemo(() => sanitizeCandles(candles), [candles]);
  const syncOverlayVisible = isSyncing;

  useEffect(() => {
    normalizedCandlesRef.current = normalizedInputCandles;
  }, [normalizedInputCandles]);

  useEffect(() => {
    loadOlderCallbackRef.current = onRequestOlderCandles ?? null;
  }, [onRequestOlderCandles]);

  useEffect(() => {
    loadingOlderRef.current = isLoadingOlder;
  }, [isLoadingOlder]);

  useEffect(() => {
    timeZoneRef.current = resolvedTimeZone;

    const options = {
      localization: {
        locale: "vi-VN",
        timeFormatter: (value: Time) => formatCrosshairTimeLabel(value, timeZoneRef.current),
      },
      timeScale: {
        tickMarkFormatter: (value: Time, tickMarkType: TickMarkType) =>
          formatAxisTimeLabel(value, timeZoneRef.current, tickMarkType),
      },
    };

    mainChartApiRef.current?.applyOptions(options);
    rsiChartApiRef.current?.applyOptions(options);
    macdChartApiRef.current?.applyOptions(options);
  }, [resolvedTimeZone]);

  useEffect(() => {
    if (!wrapperRef.current || !mainChartRef.current || !rsiChartRef.current || !macdChartRef.current) {
      return;
    }

    const width = Math.max(320, wrapperRef.current.clientWidth);

    const sharedOptions = {
      width,
      layout: {
        textColor: "rgba(220,220,235,0.9)",
        background: { type: ColorType.Solid, color: "rgba(11,6,22,0.55)" },
      },
      localization: {
        locale: "vi-VN",
        timeFormatter: (value: Time) => formatCrosshairTimeLabel(value, timeZoneRef.current),
      },
      grid: {
        vertLines: { color: "rgba(120, 95, 180, 0.16)" },
        horzLines: { color: "rgba(120, 95, 180, 0.16)" },
      },
      rightPriceScale: {
        borderColor: "rgba(136, 80, 240, 0.35)",
      },
      timeScale: {
        borderColor: "rgba(136, 80, 240, 0.35)",
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (value: Time, tickMarkType: TickMarkType) =>
          formatAxisTimeLabel(value, timeZoneRef.current, tickMarkType),
      },
      crosshair: {
        vertLine: { color: "rgba(0,255,255,0.4)", labelBackgroundColor: "#211038" },
        horzLine: { color: "rgba(140,82,255,0.45)", labelBackgroundColor: "#211038" },
      },
    };

    const mainChart = createChart(mainChartRef.current, {
      ...sharedOptions,
      height: 440,
    });
    mainChartApiRef.current = mainChart;

    const candleSeries = mainChart.addCandlestickSeries({
      upColor: "#00d8e6",
      downColor: "#ff5a8e",
      borderVisible: false,
      wickUpColor: "#00ffff",
      wickDownColor: "#ff6a99",
      priceLineColor: "#00ffff",
      lastValueVisible: true,
      priceFormat: {
        type: "price",
        precision: 6,
        minMove: 0.000001,
      },
    });
    candleSeriesRef.current = candleSeries;

    const forecastCandleSeries = mainChart.addCandlestickSeries({
      upColor: "rgba(140,82,255,0.45)",
      downColor: "rgba(140,82,255,0.45)",
      wickUpColor: "rgba(140,82,255,0.95)",
      wickDownColor: "rgba(140,82,255,0.95)",
      borderUpColor: "rgba(140,82,255,0.9)",
      borderDownColor: "rgba(140,82,255,0.9)",
      priceLineVisible: false,
      lastValueVisible: false,
      priceFormat: {
        type: "price",
        precision: 6,
        minMove: 0.000001,
      },
    });
    forecastCandleSeriesRef.current = forecastCandleSeries;

    const predictionLineSeries = mainChart.addLineSeries({
      color: "#8c52ff",
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    predictionLineSeriesRef.current = predictionLineSeries;

    const hoverHighGuideSeries = mainChart.addLineSeries({
      color: "rgba(248,193,58,0.45)",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: "#f8c13a",
      crosshairMarkerBackgroundColor: "#f8c13a",
    });
    hoverHighGuideSeriesRef.current = hoverHighGuideSeries;

    const hoverMidGuideSeries = mainChart.addLineSeries({
      color: "rgba(140,82,255,0.45)",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: "#8c52ff",
      crosshairMarkerBackgroundColor: "#8c52ff",
    });
    hoverMidGuideSeriesRef.current = hoverMidGuideSeries;

    const hoverLowGuideSeries = mainChart.addLineSeries({
      color: "rgba(52,213,255,0.45)",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: "#34d5ff",
      crosshairMarkerBackgroundColor: "#34d5ff",
    });
    hoverLowGuideSeriesRef.current = hoverLowGuideSeries;

    const rsiChart = createChart(rsiChartRef.current, {
      ...sharedOptions,
      height: 145,
    });
    rsiChartApiRef.current = rsiChart;

    const rsiSeries = rsiChart.addLineSeries({
      color: "#00ffff",
      lineWidth: 1,
      priceLineVisible: false,
    });
    rsiSeriesRef.current = rsiSeries;

    rsiSeries.createPriceLine({
      price: 70,
      color: "rgba(241,106,152,0.65)",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      axisLabelVisible: true,
      title: "RSI 70",
    });
    rsiSeries.createPriceLine({
      price: 30,
      color: "rgba(0,255,255,0.65)",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      axisLabelVisible: true,
      title: "RSI 30",
    });

    const macdChart = createChart(macdChartRef.current, {
      ...sharedOptions,
      height: 145,
    });
    macdChartApiRef.current = macdChart;

    const macdHistogram = macdChart.addHistogramSeries({
      priceLineVisible: false,
      lastValueVisible: false,
    });
    macdHistogramRef.current = macdHistogram;
    const macdSignal = macdChart.addLineSeries({
      color: "#b088f5",
      lineWidth: 1,
      priceLineVisible: false,
    });
    macdSignalRef.current = macdSignal;

    const resizeObserver = new ResizeObserver(() => {
      if (!wrapperRef.current) {
        return;
      }
      const nextWidth = wrapperRef.current.clientWidth;
      mainChartApiRef.current?.applyOptions({ width: nextWidth });
      rsiChartApiRef.current?.applyOptions({ width: nextWidth });
      macdChartApiRef.current?.applyOptions({ width: nextWidth });
    });

    resizeObserver.observe(wrapperRef.current);
    resizeObserverRef.current = resizeObserver;

    return () => {
      resizeObserverRef.current?.disconnect();
      resizeObserverRef.current = null;
      mainChart.remove();
      rsiChart.remove();
      macdChart.remove();

      mainChartApiRef.current = null;
      rsiChartApiRef.current = null;
      macdChartApiRef.current = null;
      candleSeriesRef.current = null;
      forecastCandleSeriesRef.current = null;
      predictionLineSeriesRef.current = null;
      hoverHighGuideSeriesRef.current = null;
      hoverMidGuideSeriesRef.current = null;
      hoverLowGuideSeriesRef.current = null;
      rsiSeriesRef.current = null;
      macdHistogramRef.current = null;
      macdSignalRef.current = null;
    };
  }, []);

  useEffect(() => {
    const mainChart = mainChartApiRef.current;
    if (!mainChart) {
      return;
    }

    const handleVisibleRangeChange = (range: LogicalRange | null) => {
      const currentCandles = normalizedCandlesRef.current;
      if (range && currentCandles.length > 0) {
        const lastLogicalIndex = currentCandles.length - 1;
        const movedIntoFuture = range.to > lastLogicalIndex + 0.2;
        const nearRealtime = isNearRealtime(range, currentCandles.length);
        shouldAutoFollowRef.current = nearRealtime && !movedIntoFuture;
      }

      const loadOlder = loadOlderCallbackRef.current;
      if (!loadOlder || loadingOlderRef.current || !range) {
        return;
      }

      const earlyTriggerIndex = Math.floor(Math.max(30, currentCandles.length * 0.5));
      if (range.from > earlyTriggerIndex) {
        return;
      }

      if (currentCandles.length === 0) {
        return;
      }

      const oldestTimestamp = currentCandles[0].timestamp;
      if (!oldestTimestamp || lazyLoadAnchorRef.current === oldestTimestamp) {
        return;
      }

      lazyLoadAnchorRef.current = oldestTimestamp;
      loadOlder(oldestTimestamp);
    };

    mainChart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleRangeChange);

    return () => {
      mainChart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleRangeChange);
    };
  }, []);

  useEffect(() => {
    const mainChart = mainChartApiRef.current;
    if (
      !mainChart ||
      !hoverHighGuideSeriesRef.current ||
      !hoverMidGuideSeriesRef.current ||
      !hoverLowGuideSeriesRef.current
    ) {
      return;
    }

    const clearHoverGuides = () => {
      hoverHighGuideSeriesRef.current?.setData([]);
      hoverMidGuideSeriesRef.current?.setData([]);
      hoverLowGuideSeriesRef.current?.setData([]);
    };

    const handleCrosshairMove = (param: MouseEventParams<Time>) => {
      if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
        clearHoverGuides();
        return;
      }

      const forecastPoints = forecastHoverPointsRef.current;
      if (forecastPoints.length === 0) {
        clearHoverGuides();
        return;
      }

      const hoverTime = Number(chartTimeToUnix(param.time));
      if (!Number.isFinite(hoverTime)) {
        clearHoverGuides();
        return;
      }
      const stepSeconds = Math.max(1, forecastStepSecondsRef.current);
      const firstTime = Number(forecastPoints[0].time);
      const lastTime = Number(forecastPoints[forecastPoints.length - 1].time);

      if (hoverTime < firstTime - stepSeconds || hoverTime > lastTime + stepSeconds) {
        clearHoverGuides();
        return;
      }

      const rawIndex = Math.round((hoverTime - firstTime) / stepSeconds);
      const clampedIndex = Math.max(0, Math.min(forecastPoints.length - 1, rawIndex));
      const point = forecastPoints[clampedIndex];

      if (Math.abs(hoverTime - Number(point.time)) > stepSeconds) {
        clearHoverGuides();
        return;
      }

      const halfSpan = Math.max(1, Math.floor(stepSeconds * 0.45));
      const left = (Number(point.time) - halfSpan) as UTCTimestamp;
      const right = (Number(point.time) + halfSpan) as UTCTimestamp;

      hoverHighGuideSeriesRef.current?.setData([
        { time: left, value: point.high },
        { time: right, value: point.high },
      ]);
      hoverMidGuideSeriesRef.current?.setData([
        { time: left, value: point.mid },
        { time: right, value: point.mid },
      ]);
      hoverLowGuideSeriesRef.current?.setData([
        { time: left, value: point.low },
        { time: right, value: point.low },
      ]);
    };

    mainChart.subscribeCrosshairMove(handleCrosshairMove);

    return () => {
      mainChart.unsubscribeCrosshairMove(handleCrosshairMove);
      clearHoverGuides();
    };
  }, []);

  useEffect(() => {
    lazyLoadAnchorRef.current = null;
    hasInitialFitRef.current = false;
    shouldAutoFollowRef.current = true;
    lastRenderedLengthRef.current = 0;
    lastRenderedLastTimeRef.current = Number.NaN;
    lastRenderedFirstTimeRef.current = Number.NaN;
  }, [timeframe, symbol]);

  useEffect(() => {
    if (!isLoadingOlder && normalizedInputCandles.length > 0) {
      const oldestTimestamp = normalizedInputCandles[0].timestamp;
      if (lazyLoadAnchorRef.current !== oldestTimestamp) {
        lazyLoadAnchorRef.current = null;
      }
    }
  }, [normalizedInputCandles, isLoadingOlder]);

  useEffect(() => {
    if (
      !candleSeriesRef.current ||
      !forecastCandleSeriesRef.current ||
      !predictionLineSeriesRef.current ||
      !hoverHighGuideSeriesRef.current ||
      !hoverMidGuideSeriesRef.current ||
      !hoverLowGuideSeriesRef.current ||
      !rsiSeriesRef.current ||
      !macdHistogramRef.current ||
      !macdSignalRef.current ||
      !mainChartApiRef.current ||
      !rsiChartApiRef.current ||
      !macdChartApiRef.current
    ) {
      return;
    }

    if (normalizedInputCandles.length === 0) {
      candleSeriesRef.current.setData([]);
      forecastCandleSeriesRef.current.setData([]);
      predictionLineSeriesRef.current.setData([]);
      hoverHighGuideSeriesRef.current.setData([]);
      hoverMidGuideSeriesRef.current.setData([]);
      hoverLowGuideSeriesRef.current.setData([]);
      rsiSeriesRef.current.setData([]);
      macdHistogramRef.current.setData([]);
      macdSignalRef.current.setData([]);
      forecastHoverPointsRef.current = [];
      hasInitialFitRef.current = false;
      shouldAutoFollowRef.current = true;
      lastRenderedLengthRef.current = 0;
      lastRenderedLastTimeRef.current = Number.NaN;
      lastRenderedFirstTimeRef.current = Number.NaN;
      return;
    }

    const mainRangeBeforeUpdate = mainChartApiRef.current.timeScale().getVisibleLogicalRange();
    const rsiRangeBeforeUpdate = rsiChartApiRef.current.timeScale().getVisibleLogicalRange();
    const macdRangeBeforeUpdate = macdChartApiRef.current.timeScale().getVisibleLogicalRange();

    const candleData: CandlestickData<UTCTimestamp>[] = normalizedInputCandles.map((item) => ({
      time: toUnix(item.timestamp),
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));

    candleSeriesRef.current.setData(candleData);

    if (prediction && normalizedInputCandles.length > 0 && prediction.prediction_array.length > 0) {
      const lastCandle = normalizedInputCandles[normalizedInputCandles.length - 1];
      const fallbackAnchorTime = toUnix(lastCandle.timestamp);
      const parsedAnchorMs = predictionAnchor ? parseTimestampMs(predictionAnchor.baseTimestamp) : Number.NaN;
      const hasValidAnchor = Number.isFinite(parsedAnchorMs) && Number.isFinite(predictionAnchor?.baseClose);
      const anchorTime = hasValidAnchor
        ? (Math.floor(parsedAnchorMs / 1000) as UTCTimestamp)
        : fallbackAnchorTime;
      const anchorClose = hasValidAnchor ? Number(predictionAnchor?.baseClose) : lastCandle.close;
      const stepSeconds = timeframeSeconds(timeframe);
      const forecastSeries: LineData<UTCTimestamp>[] = [{ time: anchorTime, value: anchorClose }];

      for (let index = 0; index < prediction.prediction_array.length; index += 1) {
        forecastSeries.push({
          time: (anchorTime + (index + 1) * stepSeconds) as UTCTimestamp,
          value: prediction.prediction_array[index],
        });
      }

      predictionLineSeriesRef.current.setData(forecastSeries);

      const sortedBands = [...(prediction.confidence_bands ?? [])].sort(
        (left, right) => left.quantile - right.quantile,
      );
      const lowerValues = sortedBands[0]?.values ?? [];
      const upperValues = sortedBands[sortedBands.length - 1]?.values ?? [];

      const primaryVolatilityBand = prediction.volatility_bands?.[0];
      const volatilityLowerValues = primaryVolatilityBand?.lower ?? [];
      const volatilityUpperValues = primaryVolatilityBand?.upper ?? [];

      const recentWindow = normalizedInputCandles.slice(-Math.min(120, normalizedInputCandles.length));
      const averageRecentRange =
        recentWindow.length > 0
          ? recentWindow.reduce((sum, candle) => sum + Math.max(0, candle.high - candle.low), 0) /
            recentWindow.length
          : 0;
      const anchorMagnitude = Math.max(Math.abs(anchorClose), 1);
      const wickRangeCap = Math.max(
        averageRecentRange * 3,
        anchorMagnitude * forecastRangeCapRatio(timeframe),
      );

      const forecastCandles: CandlestickData<UTCTimestamp>[] = [];
      const hoverPoints: ForecastHoverPoint[] = [];
      for (let index = 0; index < prediction.prediction_array.length; index += 1) {
        const time = (anchorTime + (index + 1) * stepSeconds) as UTCTimestamp;
        const open = index === 0 ? anchorClose : prediction.prediction_array[index - 1];
        const close = prediction.prediction_array[index];

        const confidenceLow =
          index < lowerValues.length
            ? lowerValues[index]
            : lowerValues.length > 0
              ? lowerValues[lowerValues.length - 1]
              : prediction.confidence_interval.lower;
        const confidenceHigh =
          index < upperValues.length
            ? upperValues[index]
            : upperValues.length > 0
              ? upperValues[upperValues.length - 1]
              : prediction.confidence_interval.upper;

        const volatilityLow =
          index < volatilityLowerValues.length
            ? volatilityLowerValues[index]
            : volatilityLowerValues.length > 0
              ? volatilityLowerValues[volatilityLowerValues.length - 1]
              : confidenceLow;
        const volatilityHigh =
          index < volatilityUpperValues.length
            ? volatilityUpperValues[index]
            : volatilityUpperValues.length > 0
              ? volatilityUpperValues[volatilityUpperValues.length - 1]
              : confidenceHigh;

        const fallbackHigh = Number.isFinite(volatilityHigh) ? volatilityHigh : close;
        const fallbackLow = Number.isFinite(volatilityLow) ? volatilityLow : close;
        const rangeHigh = Number.isFinite(confidenceHigh) ? confidenceHigh : fallbackHigh;
        const rangeLow = Number.isFinite(confidenceLow) ? confidenceLow : fallbackLow;

        const bodyHigh = Math.max(open, close);
        const bodyLow = Math.min(open, close);
        const rawHigh = Math.max(bodyHigh, rangeHigh);
        const rawLow = Math.min(bodyLow, rangeLow);
        const cappedHigh = Math.min(rawHigh, bodyHigh + wickRangeCap);
        const cappedLow = Math.max(rawLow, bodyLow - wickRangeCap);
        const high = Math.max(bodyHigh, cappedHigh);
        const low = Math.min(bodyLow, cappedLow);

        forecastCandles.push({ time, open, high, low, close });
        hoverPoints.push({
          time,
          low,
          mid: close,
          high,
        });
      }

      forecastCandleSeriesRef.current.setData(forecastCandles);
      hoverHighGuideSeriesRef.current.setData([]);
      hoverMidGuideSeriesRef.current.setData([]);
      hoverLowGuideSeriesRef.current.setData([]);
      forecastHoverPointsRef.current = hoverPoints;
      forecastStepSecondsRef.current = Math.max(1, stepSeconds);
    } else {
      forecastCandleSeriesRef.current.setData([]);
      predictionLineSeriesRef.current.setData([]);
      hoverHighGuideSeriesRef.current.setData([]);
      hoverMidGuideSeriesRef.current.setData([]);
      hoverLowGuideSeriesRef.current.setData([]);
      forecastHoverPointsRef.current = [];
    }

    rsiSeriesRef.current.setData(calculateRsi(normalizedInputCandles, 14));
    const macd = calculateMacd(normalizedInputCandles);
    macdHistogramRef.current.setData(macd.histogram);
    macdSignalRef.current.setData(macd.signal);

    const timeframeChanged = previousTimeframeRef.current !== timeframe;
    previousTimeframeRef.current = timeframe;
    if (timeframeChanged) {
      hasInitialFitRef.current = false;
    }

    const currentFirstTimeMs = parseTimestampMs(normalizedInputCandles[0].timestamp);
    const previousFirstTimeMs = lastRenderedFirstTimeRef.current;
    let prependShift = 0;
    if (
      Number.isFinite(previousFirstTimeMs) &&
      Number.isFinite(currentFirstTimeMs) &&
      currentFirstTimeMs < previousFirstTimeMs
    ) {
      prependShift = normalizedInputCandles.findIndex(
        (item) => parseTimestampMs(item.timestamp) === previousFirstTimeMs,
      );
      if (prependShift < 0) {
        prependShift = Math.max(0, normalizedInputCandles.length - lastRenderedLengthRef.current);
      }
    }

    if (!hasInitialFitRef.current) {
      mainChartApiRef.current.timeScale().fitContent();
      rsiChartApiRef.current.timeScale().fitContent();
      macdChartApiRef.current.timeScale().fitContent();
      hasInitialFitRef.current = true;
      shouldAutoFollowRef.current = true;
    } else {
      const latestMainRange = mainChartApiRef.current.timeScale().getVisibleLogicalRange();
      const currentlyInFuture =
        latestMainRange !== null &&
        latestMainRange.to > normalizedInputCandles.length - 1 + 0.2;

      if (
        latestMainRange &&
        (!isNearRealtime(latestMainRange, normalizedInputCandles.length) || currentlyInFuture)
      ) {
        shouldAutoFollowRef.current = false;
      }

      const latestTimeMs = parseTimestampMs(
        normalizedInputCandles[normalizedInputCandles.length - 1].timestamp,
      );
      const hasNewRightEdgeData =
        Number.isFinite(latestTimeMs) &&
        (normalizedInputCandles.length > lastRenderedLengthRef.current ||
          latestTimeMs > lastRenderedLastTimeRef.current);

      if (!loadingOlderRef.current && shouldAutoFollowRef.current && hasNewRightEdgeData && !currentlyInFuture) {
        mainChartApiRef.current.timeScale().scrollToRealTime();
        rsiChartApiRef.current.timeScale().scrollToRealTime();
        macdChartApiRef.current.timeScale().scrollToRealTime();
      } else {
        // Preserve manual viewport while realtime candles keep updating in the background.
        if (mainRangeBeforeUpdate) {
          const nextMainRange =
            prependShift > 0
              ? {
                  from: mainRangeBeforeUpdate.from + prependShift,
                  to: mainRangeBeforeUpdate.to + prependShift,
                }
              : mainRangeBeforeUpdate;
          mainChartApiRef.current.timeScale().setVisibleLogicalRange(nextMainRange);
        }
        if (rsiRangeBeforeUpdate) {
          const nextRsiRange =
            prependShift > 0
              ? {
                  from: rsiRangeBeforeUpdate.from + prependShift,
                  to: rsiRangeBeforeUpdate.to + prependShift,
                }
              : rsiRangeBeforeUpdate;
          rsiChartApiRef.current.timeScale().setVisibleLogicalRange(nextRsiRange);
        }
        if (macdRangeBeforeUpdate) {
          const nextMacdRange =
            prependShift > 0
              ? {
                  from: macdRangeBeforeUpdate.from + prependShift,
                  to: macdRangeBeforeUpdate.to + prependShift,
                }
              : macdRangeBeforeUpdate;
          macdChartApiRef.current.timeScale().setVisibleLogicalRange(nextMacdRange);
        }
      }
    }

    lastRenderedLengthRef.current = normalizedInputCandles.length;
    lastRenderedFirstTimeRef.current = currentFirstTimeMs;
    lastRenderedLastTimeRef.current = parseTimestampMs(
      normalizedInputCandles[normalizedInputCandles.length - 1].timestamp,
    );
  }, [normalizedInputCandles, timeframe, prediction, predictionAnchor]);

  const showNoDataMessage = candles.length === 0 && !syncOverlayVisible && !isPredicting;
  const showInvalidDataMessage =
    candles.length > 0 && normalizedInputCandles.length === 0 && !syncOverlayVisible && !isPredicting;

  return (
    <div ref={wrapperRef} className="relative space-y-2">
      {(syncOverlayVisible || isPredicting) && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-xl border border-cyan-300/20 bg-cosmic-900/40 backdrop-blur-[1px]">
          <div className="flex items-center gap-3 rounded-full border border-cyan-300/40 bg-cosmic-900/85 px-4 py-2 text-xs font-medium text-cyan-100">
            <span className="inline-block h-3 w-3 animate-pulse rounded-full bg-cyan-300" />
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-cyan-200/30 border-t-cyan-200" />
            <span>{isPredicting ? "Generating prediction..." : "Syncing chart data..."}</span>
          </div>
        </div>
      )}
      {(showNoDataMessage || showInvalidDataMessage) && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-xl border border-violet-400/25 bg-cosmic-900/70 px-6 text-center text-sm text-violet-200/80">
          {showNoDataMessage
            ? "Chart data is empty. Select a symbol and ensure API access token is valid."
            : "Chart data was received but contains invalid timestamp or OHLCV values. Refresh history and verify backend candle formatting."}
        </div>
      )}
      <div ref={mainChartRef} className="h-[440px] rounded-xl border border-violet-400/25" />
      <div className="flex flex-wrap items-center gap-4 px-1 text-[11px] text-violet-200/75">
        <span className="flex items-center gap-2">
          <span className="inline-block h-[2px] w-5 bg-cyan-300" />
          <span>Live candle price (realtime)</span>
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-[2px] w-5 bg-violet-400" />
          <span>Prediction (avg close)</span>
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-sm border border-violet-300/80 bg-violet-500/20" />
          <span>Forecast range (high/low)</span>
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-cyan-300" />
          <span className="inline-block h-2 w-2 rounded-full bg-violet-400" />
          <span className="inline-block h-2 w-2 rounded-full bg-amber-300" />
          <span>Hover forecast to show low/avg/high guides</span>
        </span>
      </div>
      {isLoadingOlder && (
        <div className="flex items-center gap-2 px-1 text-[11px] text-cyan-200/70">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-cyan-300/90" />
          <span>Loading older candles...</span>
        </div>
      )}
      <div className="muted-label px-1">RSI (14)</div>
      <div ref={rsiChartRef} className="h-[145px] rounded-xl border border-violet-400/25" />
      <div className="muted-label px-1">MACD (12,26,9)</div>
      <div ref={macdChartRef} className="h-[145px] rounded-xl border border-violet-400/25" />
    </div>
  );
}
