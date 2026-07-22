import { useTranslation } from "react-i18next";

import { PredictResponse, Timeframe } from "@/services/api";
import { PredictionStageDefinition, PredictionStageProgress } from "@/types/predictionProgress";
import AiCouncilPanel from "@/components/layout/AiCouncilPanel";

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
  const { t } = useTranslation();
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
    <aside className="glass-panel h-full flex flex-col overflow-hidden rounded-2xl p-4">
      <div>
        <p className="muted-label">{t("predictionPanel.title")}</p>
        <h2 className="mt-1 text-xl font-semibold text-violet-100">{t("predictionPanel.forecastTitle", { symbol })}</h2>
        <p className="mt-1 text-xs text-violet-200/75">{t("predictionPanel.subtitle")}</p>
      </div>

      <div className="scrollbar-slim mt-3 flex-1 overflow-y-auto pr-1">
        <div className="rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">{t("predictionPanel.forecastHorizon")}</p>
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
          {t("predictionPanel.active", {
            horizon: formatHoursToLabel(selectedHorizonHours),
            bars: selectedHorizonBars,
            timeframe,
          })}
        </p>
      </div>

      <button
        type="button"
        onClick={onPredict}
        disabled={loading}
        className="mt-3 w-full rounded-xl border border-cyan-300/70 bg-cyan-500/10 px-4 py-3 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading
          ? t("predictionPanel.running", {
              stage: activePredictionStage?.title ?? t("predictionPipeline.initializingTitle"),
            })
          : t("predictionPanel.generate")}
      </button>

      <div className="mb-3">
        <AiCouncilPanel symbol={symbol} timeframe={timeframe} hasPrediction={prediction !== null} />
      </div>

      {loading && predictionProgress.length > 0 && (
        <div className="mt-3 rounded-xl border border-cyan-300/35 bg-gradient-to-r from-cyan-500/10 via-violet-500/10 to-cosmic-900/65 p-3">
          <p className="muted-label">{t("predictionPanel.pipelineTitle")}</p>
          <p className="mt-1 text-xs text-cyan-100/90">
            {activePredictionStage?.description ?? t("predictionPipeline.initializingDescription")}
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
        <p className="muted-label">{t("predictionPanel.expectedMove")}</p>
        <p className={`mt-1 text-xl font-semibold ${isUp ? "price-up" : "price-down"}`}>
          {deltaPct !== null ? `${isUp ? "+" : ""}${deltaPct.toFixed(2)}%` : "--"}
        </p>
      </div>

      <div className="mt-3 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">{t("predictionPanel.confidence")}</p>
        <p className="mt-1 text-lg font-semibold text-cyan-200">
          {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : "--"}
        </p>
        <p className="mt-2 text-xs text-violet-200/80">
          {prediction && lowerAtHorizon !== null && upperAtHorizon !== null
            ? t("predictionPanel.confidenceRange", {
                lower: lowerAtHorizon.toFixed(6),
                upper: upperAtHorizon.toFixed(6),
              })
            : t("predictionPanel.confidencePlaceholder")}
        </p>
      </div>

      <div className="mt-4 rounded-xl border border-violet-400/25 bg-cosmic-900/60 p-3">
        <p className="muted-label">{t("predictionPanel.sentimentScore")}</p>
        <p className="mt-1 text-lg font-semibold text-cyan-200">
          {prediction ? prediction.sentiment_score.toFixed(3) : "--"}
        </p>
        <p className="mt-1 text-xs text-violet-200/80">
          {prediction
            ? t("predictionPanel.sentimentSource", {
                source: prediction.sentiment_source,
                timeframe: prediction.timeframe,
              })
            : t("predictionPanel.sentimentPlaceholder")}
        </p>
        <p className="mt-1 text-xs text-violet-200/80">
          {prediction
            ? t("predictionPanel.externalSentiment", {
                score: prediction.external_sentiment_score.toFixed(3),
                source: prediction.external_sentiment_source,
              })
            : t("predictionPanel.externalSentimentPlaceholder")}
        </p>
      </div>
      </div>

    </aside>
  );
}
