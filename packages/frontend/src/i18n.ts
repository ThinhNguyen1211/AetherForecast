import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const resources = {
  en: {
    translation: {
      aiCouncil: {
        title: "AI AGENT TEAM",
        analyzeButton: "Analyze & Generate Signals",
        analyzing: "Agents are analyzing strategy...",
        needPrediction: "Run ML Prediction first to unlock this feature.",
        councilReasoning: "Council Reasoning",
        hideTranscript: "Hide agent transcript",
        viewTranscript: "View full agent transcript",
        actionLabels: {
          LONG: "🟢 LONG",
          SHORT: "🔴 SHORT",
          HOLD: "⏸ HOLD",
        },
        confidence: "Confidence",
        entryZone: "Entry Zone & Timing",
        leverage: "Leverage",
        positionSize: "Pos Size",
        takeProfit1: "Take Profit 1",
        takeProfit2: "Take Profit 2",
        tpSafe: "TP 1 (Safe)",
        tpMoon: "TP 2 (Moon)",
        stopLoss: "Hard Stop Loss",
        riskReward: "Risk/Reward (TP1)",
        invalidationPoint: "⚠️ Invalidation Point:",
      },
      language: {
        en: "EN",
        vi: "VI",
      },
    },
  },
  vi: {
    translation: {
      aiCouncil: {
        title: "ĐỘI NGŨ AI",
        analyzeButton: "Phân tích & Lên Tín hiệu",
        analyzing: "Các tác nhân AI đang phân tích chiến lược...",
        needPrediction: "Chạy Dự đoán ML trước để mở khóa tính năng này.",
        councilReasoning: "Lập luận của Hội đồng",
        hideTranscript: "Ẩn bản ghi tác nhân",
        viewTranscript: "Xem toàn bộ bản ghi tác nhân",
        actionLabels: {
          LONG: "🟢 MUA",
          SHORT: "🔴 BÁN",
          HOLD: "⏸ Đứng ngoài",
        },
        confidence: "Độ tin cậy",
        entryZone: "Vùng vào lệnh & Thời điểm",
        leverage: "Đòn bẩy",
        positionSize: "Kích thước vị thế",
        takeProfit1: "Chốt lời 1",
        takeProfit2: "Chốt lời 2",
        tpSafe: "CL 1 (An toàn)",
        tpMoon: "CL 2 (Mặt trăng)",
        stopLoss: "Cắt lỗ cứng",
        riskReward: "Rủi ro/Lợi nhuận (CL1)",
        invalidationPoint: "⚠️ Điểm vô hiệu hóa:",
      },
      language: {
        en: "EN",
        vi: "VI",
      },
    },
  },
};

i18n.use(initReactI18next).init({
  resources,
  lng: "vi",
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
  },
});

export default i18n;
