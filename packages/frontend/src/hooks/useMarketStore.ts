import { create } from "zustand";

import {
  Candle,
  PredictResponse,
  Timeframe,
  clearAuthToken,
  getAuthToken,
  saveAuthToken,
} from "@/services/api";

interface MarketState {
  token: string;
  symbol: string;
  timeframe: Timeframe;
  symbols: string[];
  chartCandles: Candle[];
  prediction: PredictResponse | null;
  loadingSymbols: boolean;
  loadingChart: boolean;
  loadingPrediction: boolean;
  apiStatus: "idle" | "online" | "offline";
  wsStatus: "idle" | "connecting" | "online" | "offline";
  errorMessage: string;
  setToken: (token: string) => void;
  clearToken: () => void;
  setSymbol: (symbol: string) => void;
  setTimeframe: (timeframe: Timeframe) => void;
  setSymbols: (symbols: string[]) => void;
  setChartCandles: (candles: Candle[] | ((prev: Candle[]) => Candle[])) => void;
  setPrediction: (prediction: PredictResponse | null) => void;
  setLoadingSymbols: (value: boolean) => void;
  setLoadingChart: (value: boolean) => void;
  setLoadingPrediction: (value: boolean) => void;
  setApiStatus: (status: "idle" | "online" | "offline") => void;
  setWsStatus: (status: "idle" | "connecting" | "online" | "offline") => void;
  setErrorMessage: (message: string) => void;
}

const initialToken = getAuthToken();

export const useMarketStore = create<MarketState>((set) => ({
  token: initialToken,
  symbol: "BTCUSDT",
  timeframe: "1h",
  symbols: [],
  chartCandles: [],
  prediction: null,
  loadingSymbols: false,
  loadingChart: false,
  loadingPrediction: false,
  apiStatus: "idle",
  wsStatus: "idle",
  errorMessage: "",
  setToken: (token) => {
    saveAuthToken(token);
    set({ token });
  },
  clearToken: () => {
    clearAuthToken();
    set({ token: "" });
  },
  setSymbol: (symbol) => set({ symbol }),
  setTimeframe: (timeframe) => set({ timeframe }),
  setSymbols: (symbols) => set({ symbols }),
  setChartCandles: (candles) =>
    set((state) => ({
      chartCandles: typeof candles === "function" ? candles(state.chartCandles) : candles,
    })),
  setPrediction: (prediction) => set({ prediction }),
  setLoadingSymbols: (loadingSymbols) => set({ loadingSymbols }),
  setLoadingChart: (loadingChart) => set({ loadingChart }),
  setLoadingPrediction: (loadingPrediction) => set({ loadingPrediction }),
  setApiStatus: (apiStatus) => set({ apiStatus }),
  setWsStatus: (wsStatus) => set({ wsStatus }),
  setErrorMessage: (errorMessage) => set({ errorMessage }),
}));
