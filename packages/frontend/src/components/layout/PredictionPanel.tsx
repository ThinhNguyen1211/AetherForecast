import { PredictResponse, Timeframe } from "@/services/api";

interface HorizonOption {
  label: string;
  hours: number;
}

interface PredictionPanelProps {
  symbol: string;
  timeframe: Timeframe;
  lastPrice: number | null;
  lastCandleTimestamp: string | null;
  prediction: PredictResponse | null;
  horizonOptions: HorizonOption[];
  selectedHorizonHours: number;
  selectedHorizonBars: number;
  loading: boolean;
  onSelectHorizon: (hours: number) => void;
  onPredict: () => void;
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

function formatHoursToLabel(hours: number): string {
  if (hours % 24 === 0) {
    return `${hours / 24}d`;
  }
  return `${hours}h`;
}

export default function PredictionPanel({
  symbol,
  timeframe,
  lastPrice,
  lastCandleTimestamp,
  prediction,
  horizonOptions,
  selectedHorizonHours,
  selectedHorizonBars,
  loading,
  onSelectHorizon,
  onPredict,
}: PredictionPanelProps) {
  const predictedPrice = prediction?.predicted_price ?? null;
  const horizonTargetPrice =
    prediction?.prediction_array[prediction.prediction_array.length - 1] ?? predictedPrice;

  const effectiveBars = prediction?.horizon ?? selectedHorizonBars;
  const effectiveHours = (effectiveBars * timeframeSeconds(timeframe)) / 3600;
  const targetTimestampMs =
    lastCandleTimestamp !== null
      ? Date.parse(lastCandleTimestamp) + effectiveBars * timeframeSeconds(timeframe) * 1000
      : Number.NaN;
  const targetTimeLabel = Number.isFinite(targetTimestampMs)
    ? new Date(targetTimestampMs).toLocaleString()
    : "--";

  const sortedBands = prediction?.confidence_bands
    ? [...prediction.confidence_bands].sort((left, right) => left.quantile - right.quantile)
    : [];
  const lowerBand = sortedBands[0];
  const upperBand = sortedBands[sortedBands.length - 1];
  const lowerAtHorizon =
    lowerBand?.values[Math.min(Math.max(effectiveBars - 1, 0), lowerBand.values.length - 1)] ??
    prediction?.confidence_interval.lower ??
    null;
  const upperAtHorizon =
    upperBand?.values[Math.min(Math.max(effectiveBars - 1, 0), upperBand.values.length - 1)] ??
    prediction?.confidence_interval.upper ??
    null;

  const lowerNextBar = lowerBand?.values[0] ?? prediction?.confidence_interval.lower ?? null;
  const upperNextBar = upperBand?.values[0] ?? prediction?.confidence_interval.upper ?? null;
  const avgNextBar =
    predictedPrice !== null && lowerNextBar !== null && upperNextBar !== null
      ? (predictedPrice + lowerNextBar + upperNextBar) / 3
      : predictedPrice;
  const horizonAverage =
    prediction && prediction.prediction_array.length > 0
      ? prediction.prediction_array.reduce((sum, value) => sum + value, 0) / prediction.prediction_array.length
      : null;

  const deltaPct =
    horizonTargetPrice !== null && lastPrice !== null && lastPrice > 0
      ? ((horizonTargetPrice - lastPrice) / lastPrice) * 100
      : null;

  const isUp = (deltaPct ?? 0) >= 0;

  return (
    <section className="glass-panel scrollbar-slim flex h-[72vh] min-h-[28rem] flex-col overflow-y-auto rounded-2xl p-4 lg:h-[calc(100vh-10.5rem)]">
      <div>
        <p className="muted-label">Prediction</p>
        <h2 className="mt-1 text-xl font-semibold text-violet-100">{symbol} Forecast</h2>
        <p className="mt-1 text-xs text-violet-200/75">Horizon controls use hours/days and map to chart timeframe steps.</p>
      </div>

      <div className="mt-4 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">Forecast Horizon</p>
        <div className="mt-2 grid grid-cols-2 gap-2">
          {horizonOptions.map((option) => (
            <button
              key={option.label}
              type="button"
              onClick={() => onSelectHorizon(option.hours)}
              className={`rounded-lg border px-3 py-2 text-xs font-semibold transition ${
                selectedHorizonHours === option.hours
                  ? "border-cyan-300/80 bg-cyan-400/15 text-cyan-100"
                  : "border-violet-400/35 bg-cosmic-900/70 text-violet-200 hover:border-violet-300/60"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-violet-200/75">
          Active: {formatHoursToLabel(selectedHorizonHours)} requested, {selectedHorizonBars} steps on {timeframe}.
        </p>
      </div>

      <button
        type="button"
        onClick={onPredict}
        disabled={loading}
        className="mt-3 rounded-xl border border-cyan-300/70 bg-cyan-500/10 px-4 py-3 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Generating forecast..." : "Generate prediction"}
      </button>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
          <p className="muted-label">Current</p>
          <p className="mt-1 font-mono text-lg text-violet-50">
            {lastPrice !== null ? lastPrice.toFixed(6) : "--"}
          </p>
        </div>
        <div className="rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
          <p className="muted-label">Next Step Forecast</p>
          <p className="mt-1 font-mono text-lg text-violet-50">
            {predictedPrice !== null ? predictedPrice.toFixed(6) : "--"}
          </p>
        </div>
      </div>

      <div className="mt-3 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">Horizon Target</p>
        <p className="mt-1 font-mono text-lg text-violet-50">
          {horizonTargetPrice !== null ? horizonTargetPrice.toFixed(6) : "--"}
        </p>
        <p className="mt-1 text-xs text-violet-200/80">Target time: {targetTimeLabel}</p>
        <p className="mt-1 text-xs text-violet-200/80">
          Model horizon: {effectiveBars} steps (~{formatHoursToLabel(Math.max(1, Math.round(effectiveHours)))}).
        </p>
      </div>

      <div className="mt-3 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">Expected move</p>
        <p className={`mt-1 text-xl font-semibold ${isUp ? "price-up" : "price-down"}`}>
          {deltaPct !== null ? `${isUp ? "+" : ""}${deltaPct.toFixed(2)}%` : "--"}
        </p>
      </div>

      <div className="mt-3 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">Confidence</p>
        <p className="mt-1 text-lg font-semibold text-cyan-200">
          {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : "--"}
        </p>
        <p className="mt-2 text-xs text-violet-200/80">
          {prediction && lowerAtHorizon !== null && upperAtHorizon !== null
            ? `${lowerAtHorizon.toFixed(6)} - ${upperAtHorizon.toFixed(6)} (horizon target)`
            : "Run a prediction to view confidence interval."}
        </p>
      </div>

      <div className="mt-3 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">Forecast Candle ({timeframe})</p>
        <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
          <div className="rounded-md border border-violet-400/20 bg-cosmic-900/70 p-2">
            <p className="text-violet-200/75">Low</p>
            <p className="mt-1 font-mono text-violet-50">
              {lowerNextBar !== null ? lowerNextBar.toFixed(6) : "--"}
            </p>
          </div>
          <div className="rounded-md border border-violet-400/20 bg-cosmic-900/70 p-2">
            <p className="text-violet-200/75">Avg</p>
            <p className="mt-1 font-mono text-violet-50">
              {avgNextBar !== null ? avgNextBar.toFixed(6) : "--"}
            </p>
          </div>
          <div className="rounded-md border border-violet-400/20 bg-cosmic-900/70 p-2">
            <p className="text-violet-200/75">High</p>
            <p className="mt-1 font-mono text-violet-50">
              {upperNextBar !== null ? upperNextBar.toFixed(6) : "--"}
            </p>
          </div>
        </div>
        <p className="mt-2 text-xs text-violet-200/80">
          {horizonAverage !== null
            ? `Horizon average close: ${horizonAverage.toFixed(6)}`
            : "Run prediction to view low/avg/high per forecast candle."}
        </p>
      </div>

      <div className="mt-4 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">Sentiment Score (Auto)</p>
        <p className="mt-1 text-lg font-semibold text-cyan-200">
          {prediction ? prediction.sentiment_score.toFixed(3) : "--"}
        </p>
        <p className="mt-1 text-xs text-violet-200/80">
          {prediction
            ? `Source: ${prediction.sentiment_source} (${prediction.timeframe})`
            : "Sentiment is computed automatically from latest market candles."}
        </p>
        <p className="mt-1 text-xs text-violet-200/80">
          {prediction
            ? `External: ${prediction.external_sentiment_score.toFixed(3)} (${prediction.external_sentiment_source})`
            : "External sentiment is fetched from live macro/news/social feeds per prediction."}
        </p>
      </div>

      <details className="mt-4 rounded-xl border border-violet-400/20 bg-cosmic-900/45 p-3">
        <summary className="cursor-pointer text-sm font-semibold text-violet-100/90">Model Note</summary>
        <p className="mt-2 text-sm leading-6 text-violet-100/80">
          {prediction
            ? `Forecast blends live candles, candlestick patterns, volatility context, and external sentiment. Expected move is projected over ${effectiveBars} bars with current confidence ${(prediction.confidence * 100).toFixed(1)}%.`
            : "Run a prediction to see an expanded model note."}
        </p>
      </details>
    </section>
  );
}
