import axios from "axios";

import { CognitoSignInResult, refreshCognitoSession } from "@/services/cognito-auth";

export const TOKEN_STORAGE_KEY = "aetherforecast.jwt";
export const SESSION_STORAGE_KEY = "aetherforecast.session";
const TOKEN_REFRESH_WINDOW_MS = 90_000;

export interface AuthSession {
  idToken: string;
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  expiresAt: number;
}

export type Timeframe = "1m" | "5m" | "15m" | "1h" | "4h" | "1d" | "1w";

export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictRequest {
  symbol: string;
  timeframe: Timeframe;
  latest_candles: Candle[];
  horizon: number;
}

export interface ConfidenceBand {
  quantile: number;
  values: number[];
}

export interface PredictResponse {
  symbol: string;
  timeframe: Timeframe;
  horizon: number;
  predicted_price: number;
  prediction_array: number[];
  confidence: number;
  sentiment_score: number;
  sentiment_source: string;
  external_sentiment_score: number;
  external_sentiment_source: string;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  confidence_bands: ConfidenceBand[];
  trend_direction: "up" | "down" | "flat";
  model_name: string;
  model_version: string;
  explanation: string;
  generated_at: string;
}

export interface ChartResponse {
  symbol: string;
  timeframe: Timeframe;
  candles: Candle[];
}

interface Binance24hTickerPayload {
  symbol: string;
  lastPrice: string;
  priceChangePercent: string;
}

export interface Binance24hTicker {
  symbol: string;
  lastPrice: number;
  changePercent: number;
}

export interface RealtimeKlineMessage {
  event: "kline";
  symbol: string;
  timeframe: Timeframe;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  is_closed: boolean;
}

let refreshInFlight: Promise<AuthSession | null> | null = null;

function hasLocalStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function parseAuthSession(raw: string | null): AuthSession | null {
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<AuthSession>;
    const idToken = typeof parsed.idToken === "string" ? parsed.idToken.trim() : "";
    const accessToken = typeof parsed.accessToken === "string" ? parsed.accessToken.trim() : "";
    const refreshToken = typeof parsed.refreshToken === "string" ? parsed.refreshToken.trim() : "";
    const expiresIn = Number(parsed.expiresIn);
    const expiresAt = Number(parsed.expiresAt);

    if (!idToken || !Number.isFinite(expiresAt)) {
      return null;
    }

    return {
      idToken,
      accessToken,
      refreshToken,
      expiresIn: Number.isFinite(expiresIn) ? expiresIn : 3600,
      expiresAt,
    };
  } catch {
    return null;
  }
}

function toAuthSession(session: CognitoSignInResult, fallbackRefreshToken = ""): AuthSession {
  const expiresIn = Math.max(60, Number(session.expiresIn) || 3600);
  const refreshToken = (session.refreshToken || fallbackRefreshToken).trim();

  return {
    idToken: session.idToken,
    accessToken: session.accessToken,
    refreshToken,
    expiresIn,
    expiresAt: Date.now() + expiresIn * 1000,
  };
}

export function saveAuthSession(session: CognitoSignInResult, fallbackRefreshToken = ""): AuthSession | null {
  if (!hasLocalStorage()) {
    return null;
  }

  const persisted = toAuthSession(session, fallbackRefreshToken);
  localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(persisted));
  localStorage.setItem(TOKEN_STORAGE_KEY, persisted.idToken);
  return persisted;
}

export function getAuthSession(): AuthSession | null {
  if (!hasLocalStorage()) {
    return null;
  }

  const parsed = parseAuthSession(localStorage.getItem(SESSION_STORAGE_KEY));
  if (parsed) {
    return parsed;
  }

  const legacyToken = localStorage.getItem(TOKEN_STORAGE_KEY)?.trim();
  if (!legacyToken) {
    return null;
  }

  return {
    idToken: legacyToken,
    accessToken: "",
    refreshToken: "",
    expiresIn: 0,
    expiresAt: Number.MAX_SAFE_INTEGER,
  };
}

function shouldRefreshSession(session: AuthSession): boolean {
  if (!session.refreshToken) {
    return false;
  }
  return Date.now() + TOKEN_REFRESH_WINDOW_MS >= session.expiresAt;
}

async function ensureFreshSession(): Promise<AuthSession | null> {
  const session = getAuthSession();
  if (!session) {
    return null;
  }

  if (!shouldRefreshSession(session)) {
    return session;
  }

  if (refreshInFlight) {
    return refreshInFlight;
  }

  refreshInFlight = (async () => {
    try {
      const refreshed = await refreshCognitoSession(session.refreshToken);
      return saveAuthSession(refreshed, session.refreshToken);
    } catch {
      clearAuthToken();
      return null;
    } finally {
      refreshInFlight = null;
    }
  })();

  return refreshInFlight;
}

function resolveApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) {
    return configured;
  }

  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      return "http://localhost:8000";
    }

    return `${window.location.protocol}//${window.location.hostname}`;
  }

  return "http://localhost:8000";
}

const api = axios.create({
  baseURL: resolveApiBaseUrl(),
  timeout: 15000,
});

api.interceptors.request.use(async (config) => {
  const session = await ensureFreshSession();
  const token = session?.idToken ?? getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export async function fetchSymbols(): Promise<string[]> {
  const response = await api.get<{ symbols: string[] }>("/symbols");
  return response.data.symbols ?? [];
}

export async function fetchChart(
  symbol: string,
  timeframe: Timeframe,
  limit = 1200,
  fromTimestamp?: string,
): Promise<Candle[]> {
  const response = await api.get<ChartResponse>(`/chart/${symbol}`, {
    params: {
      timeframe,
      limit,
      ...(fromTimestamp ? { from_timestamp: fromTimestamp } : {}),
    },
  });
  return response.data.candles ?? [];
}

export async function fetchBinance24hTicker(symbol: string): Promise<Binance24hTicker> {
  const response = await axios.get<Binance24hTickerPayload>("https://api.binance.com/api/v3/ticker/24hr", {
    params: { symbol: symbol.toUpperCase() },
    timeout: 8000,
  });

  const lastPrice = Number(response.data.lastPrice);
  const changePercent = Number(response.data.priceChangePercent);
  if (!Number.isFinite(lastPrice) || !Number.isFinite(changePercent)) {
    throw new Error("Invalid Binance 24h ticker payload");
  }

  return {
    symbol: response.data.symbol,
    lastPrice,
    changePercent,
  };
}

export async function fetchPrediction(input: PredictRequest): Promise<PredictResponse> {
  const response = await api.post<PredictResponse>("/predict", input);
  return response.data;
}

export async function fetchHealth(): Promise<{ status: string }> {
  const response = await api.get<{ status: string }>("/health");
  return response.data;
}

export function saveAuthToken(token: string): void {
  if (!hasLocalStorage()) {
    return;
  }

  const normalized = token.trim();
  if (!normalized) {
    return;
  }

  localStorage.setItem(TOKEN_STORAGE_KEY, normalized);

  const session = parseAuthSession(localStorage.getItem(SESSION_STORAGE_KEY));
  if (session) {
    session.idToken = normalized;
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
  }
}

export function clearAuthToken(): void {
  if (!hasLocalStorage()) {
    return;
  }

  localStorage.removeItem(TOKEN_STORAGE_KEY);
  localStorage.removeItem(SESSION_STORAGE_KEY);
}

export function getAuthToken(): string {
  if (!hasLocalStorage()) {
    return "";
  }

  const sessionToken = parseAuthSession(localStorage.getItem(SESSION_STORAGE_KEY))?.idToken;
  if (sessionToken) {
    return sessionToken;
  }

  return localStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
}

function resolveWebSocketBaseUrl(): string {
  const apiBase = resolveApiBaseUrl();
  if (apiBase.startsWith("https://")) {
    return apiBase.replace("https://", "wss://");
  }
  if (apiBase.startsWith("http://")) {
    return apiBase.replace("http://", "ws://");
  }
  return apiBase;
}

export function connectRealtimeKline(
  symbol: string,
  timeframe: Timeframe,
  onMessage: (message: RealtimeKlineMessage) => void,
  onStatusChange?: (status: "connecting" | "online" | "offline") => void,
): WebSocket {
  const wsBase = resolveWebSocketBaseUrl();
  const url = `${wsBase}/ws/${symbol.toUpperCase()}?timeframe=${timeframe}`;
  const socket = new WebSocket(url);

  onStatusChange?.("connecting");

  socket.onopen = () => {
    onStatusChange?.("online");
  };

  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data) as Partial<RealtimeKlineMessage>;
      if (payload.event !== "kline") {
        return;
      }

      if (!payload.symbol || !payload.timeframe || !payload.timestamp) {
        return;
      }

      onMessage(payload as RealtimeKlineMessage);
    } catch {
      // Ignore malformed frames.
    }
  };

  socket.onerror = () => {
    onStatusChange?.("offline");
  };

  socket.onclose = () => {
    onStatusChange?.("offline");
  };

  return socket;
}
