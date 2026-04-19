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

export const PREDICTION_PIPELINE_STEPS: PredictionStageDefinition[] = [
  {
    key: "sync_market_data",
    title: "Đồng bộ nến mới nhất",
    description: "Lấy dữ liệu chart mới nhất để tránh dự đoán trên snapshot cũ.",
  },
  {
    key: "fetch_external_context",
    title: "Thu thập ngữ cảnh thị trường",
    description: "Chuẩn bị sentiment và bối cảnh ngoài giá trong backend.",
  },
  {
    key: "prepare_payload",
    title: "Chuẩn hóa dữ liệu đầu vào",
    description: "Gom candles và horizon thành payload thống nhất cho model.",
  },
  {
    key: "send_request",
    title: "Gửi yêu cầu dự đoán",
    description: "Gửi payload tới endpoint /predict.",
  },
  {
    key: "run_inference",
    title: "Model đang suy luận",
    description: "Tính forecast array, confidence bands và biến động.",
  },
  {
    key: "render_output",
    title: "Xuất kết quả lên biểu đồ",
    description: "Vẽ prediction line, vùng dự báo và cập nhật panel.",
  },
];
