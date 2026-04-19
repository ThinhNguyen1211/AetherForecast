import { PredictResponse, Timeframe } from "@/services/api";
import { PredictionStageDefinition, PredictionStageProgress } from "@/types/predictionProgress";

interface HorizonOption {
  label: string;
  hours: number;
}

interface PredictionPanelProps {
  symbol: string;
  timeframe: Timeframe;
  lastPrice: number | null;
  prediction: PredictResponse | null;
  horizonOptions: HorizonOption[];
  selectedHorizonHours: number;
  selectedHorizonBars: number;
  loading: boolean;
  predictionProgress?: PredictionStageProgress[];
  activePredictionStage?: PredictionStageDefinition | null;
  onSelectHorizon: (hours: number) => void;
  onPredict: () => void;
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
  prediction,
  horizonOptions,
  selectedHorizonHours,
  selectedHorizonBars,
  loading,
  predictionProgress = [],
  activePredictionStage = null,
  onSelectHorizon,
  onPredict,
}: PredictionPanelProps) {
  const horizonTargetPrice =
    prediction?.prediction_array[prediction.prediction_array.length - 1] ?? prediction?.predicted_price ?? null;

  const effectiveBars = prediction?.horizon ?? selectedHorizonBars;

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
        {loading
          ? `Đang chạy: ${activePredictionStage?.title ?? "Tạo dự đoán"}`
          : "Generate prediction"}
      </button>

      {loading && predictionProgress.length > 0 && (
        <div className="mt-3 rounded-xl border border-cyan-300/35 bg-gradient-to-r from-cyan-500/10 via-violet-500/10 to-cosmic-900/65 p-3">
          <p className="muted-label">Prediction Pipeline</p>
          <p className="mt-1 text-xs text-cyan-100/90">
            {activePredictionStage?.description ?? "Đang khởi tạo luồng dự đoán."}
          </p>
          <div className="mt-3 space-y-2">
            {predictionProgress.map((step, index) => (
              <div
                key={step.key}
                className="grid grid-cols-[auto_1fr] items-start gap-x-2 gap-y-0.5 text-[11px]"
              >
                <span
                  className={`mt-1 inline-block h-2.5 w-2.5 rounded-full ${
                    step.status === "done"
                      ? "bg-cyan-300"
                      : step.status === "active"
                        ? "animate-pulse bg-violet-300"
                        : "bg-violet-200/30"
                  }`}
                />
                <div>
                  <p
                    className={`font-semibold ${
                      step.status === "pending" ? "text-violet-200/65" : "text-cyan-100"
                    }`}
                  >
                    {index + 1}. {step.title}
                  </p>
                  <p className="text-violet-100/70">{step.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

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

    </section>
  );
}
