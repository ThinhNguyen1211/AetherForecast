import { useTranslation } from "react-i18next";

export type PredictionStageKey =
  | "sync_market_data"
  | "fetch_external_context"
  | "prepare_payload"
  | "send_request"
  | "run_inference"
  | "render_output";

export type PredictionStageStatus = "done" | "active" | "pending";

export interface PredictionStageDefinition {
  key: PredictionStageKey;
  title: string;
  description: string;
}

export interface PredictionStageProgress extends PredictionStageDefinition {
  status: PredictionStageStatus;
}

export const PREDICTION_PIPELINE_STAGE_KEYS: PredictionStageKey[] = [
  "sync_market_data",
  "fetch_external_context",
  "prepare_payload",
  "send_request",
  "run_inference",
  "render_output",
];

export function usePredictionPipelineSteps(): PredictionStageDefinition[] {
  const { t } = useTranslation();

  return PREDICTION_PIPELINE_STAGE_KEYS.map((key) => ({
    key,
    title: t(`predictionPipeline.stages.${key}.title`),
    description: t(`predictionPipeline.stages.${key}.description`),
  }));
}
