from datetime import datetime, timezone
from functools import lru_cache
import logging
import math
import re
import time
from statistics import pstdev
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.core.config import get_settings
from src.data.external_covariates import build_external_covariate_signal
from src.data.sentiment import SentimentScorer
from src.ml.model_loader import LoadedForecastModel, get_loaded_forecasting_model
from src.ml.schemas import (
    ConfidenceBand,
    ConfidenceInterval,
    PatternMarker,
    PredictRequest,
    PredictResponse,
    VolatilityBand,
)

logger = logging.getLogger(__name__)
NUMERIC_REGEX = re.compile(r"-?\d+(?:\.\d+)?")

try:
    import talib
except Exception:  # pragma: no cover
    talib = None


class ForecastInferenceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.sentiment_scorer = SentimentScorer(
            mode=self.settings.sentiment_mode,
            model_id=self.settings.sentiment_model_id,
            cache_dir=self.settings.sentiment_cache_dir,
            external_enabled=self.settings.external_sentiment_enabled,
            external_refresh_seconds=self.settings.external_sentiment_refresh_seconds,
            news_rss_urls=self.settings.external_sentiment_news_rss_urls,
            news_api_endpoint=self.settings.external_news_api_endpoint,
            news_api_key=self.settings.external_news_api_key,
            news_api_query=self.settings.external_news_api_query,
            news_api_max_items=self.settings.external_news_api_limit,
            x_sentiment_endpoint=self.settings.external_x_sentiment_endpoint,
            x_search_endpoint=self.settings.external_x_search_endpoint,
            x_search_bearer_token=self.settings.external_x_search_bearer_token,
            x_search_query=self.settings.external_x_search_query,
            x_search_max_items=self.settings.external_x_search_limit,
            geopolitical_sentiment_endpoint=self.settings.external_geopolitical_sentiment_endpoint,
            event_keywords=self.settings.external_event_keywords,
        )

    def _load_model(self) -> LoadedForecastModel:
        model_s3_uri = self.settings.model_s3_uri
        if not model_s3_uri.startswith("s3://"):
            raise RuntimeError("MODEL_S3_URI must be an s3:// URI to resolve manifest/latest.json")

        return get_loaded_forecasting_model(
            model_s3_uri=model_s3_uri,
            fallback_model_id=self.settings.hf_model_fallback_id,
            tokenizer_fallback_id=self.settings.hf_tokenizer_fallback_id,
            model_cache_dir=self.settings.model_cache_dir,
            aws_region=self.settings.aws_region,
            endpoint_url=self.settings.aws_endpoint_url,
            torch_dtype=self.settings.model_torch_dtype,
            require_s3_model=True,
        )

    def _apply_sentiment(self, forecast: np.ndarray, sentiment_score: float | None) -> np.ndarray:
        if sentiment_score is None:
            return forecast

        bounded = max(-1.0, min(1.0, sentiment_score))
        adjustment = 1.0 + bounded * 0.002
        return forecast * adjustment

    def _pattern_signal_series(self, request: PredictRequest) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        opens = np.asarray([float(item.open) for item in request.latest_candles], dtype=np.float64)
        highs = np.asarray([float(item.high) for item in request.latest_candles], dtype=np.float64)
        lows = np.asarray([float(item.low) for item in request.latest_candles], dtype=np.float64)
        closes = np.asarray([float(item.close) for item in request.latest_candles], dtype=np.float64)

        if closes.size == 0:
            return np.asarray([], dtype=np.float64), {}

        patterns: dict[str, np.ndarray] = {}

        if talib is not None:
            def _safe_talib_pattern(name: str) -> np.ndarray | None:
                function = getattr(talib, name, None)
                if function is None:
                    return None

                try:
                    return np.tanh(function(opens, highs, lows, closes).astype(np.float64) / 100.0)
                except Exception:
                    return None

            try:
                for output_name, talib_name in (
                    ("doji", "CDLDOJI"),
                    ("engulfing", "CDLENGULFING"),
                    ("harami", "CDLHARAMI"),
                    ("piercing", "CDLPIERCING"),
                    ("dark_cloud_cover", "CDLDARKCLOUDCOVER"),
                    ("hammer", "CDLHAMMER"),
                    ("inverted_hammer", "CDLINVERTEDHAMMER"),
                    ("shooting_star", "CDLSHOOTINGSTAR"),
                    ("spinning_top", "CDLSPINNINGTOP"),
                    ("dragonfly_doji", "CDLDRAGONFLYDOJI"),
                    ("gravestone_doji", "CDLGRAVESTONEDOJI"),
                    ("morning_star", "CDLMORNINGSTAR"),
                    ("evening_star", "CDLEVENINGSTAR"),
                    ("hanging_man", "CDLHANGINGMAN"),
                    ("three_white_soldiers", "CDL3WHITESOLDIERS"),
                    ("three_black_crows", "CDL3BLACKCROWS"),
                    ("marubozu", "CDLMARUBOZU"),
                ):
                    value = _safe_talib_pattern(talib_name)
                    if value is not None:
                        patterns[output_name] = value
            except Exception as exc:
                logger.warning("TA-Lib pattern extraction failed in inference: %s", exc)

        if not patterns:
            candle_range = np.maximum(highs - lows, 1e-8)
            body = np.abs(closes - opens)
            upper_shadow = highs - np.maximum(opens, closes)
            lower_shadow = np.minimum(opens, closes) - lows

            doji = np.where(body / candle_range < 0.1, 1.0, 0.0)
            hammer = np.where(
                (lower_shadow / candle_range > 0.55) & (upper_shadow / candle_range < 0.2),
                np.sign(closes - opens),
                0.0,
            )
            shooting_star = np.where(
                (upper_shadow / candle_range > 0.55) & (lower_shadow / candle_range < 0.2),
                -np.sign(closes - opens),
                0.0,
            )
            inverted_hammer = np.where(
                (upper_shadow / candle_range > 0.55) & (lower_shadow / candle_range < 0.22),
                np.sign(closes - opens),
                0.0,
            )
            spinning_top = np.where(
                (body / candle_range < 0.35)
                & (upper_shadow / candle_range > 0.25)
                & (lower_shadow / candle_range > 0.25),
                np.sign(closes - opens) * 0.5,
                0.0,
            )
            dragonfly_doji = np.where(
                (body / candle_range < 0.10)
                & (lower_shadow / candle_range > 0.60)
                & (upper_shadow / candle_range < 0.12),
                1.0,
                0.0,
            )
            gravestone_doji = np.where(
                (body / candle_range < 0.10)
                & (upper_shadow / candle_range > 0.60)
                & (lower_shadow / candle_range < 0.12),
                -1.0,
                0.0,
            )

            prev_open = np.roll(opens, 1)
            prev_close = np.roll(closes, 1)
            prev_high = np.roll(highs, 1)
            prev_low = np.roll(lows, 1)
            bullish_engulf = (
                (closes > opens)
                & (prev_close < prev_open)
                & (closes >= prev_open)
                & (opens <= prev_close)
            )
            bearish_engulf = (
                (closes < opens)
                & (prev_close > prev_open)
                & (opens >= prev_close)
                & (closes <= prev_open)
            )
            engulfing = bullish_engulf.astype(np.float64) - bearish_engulf.astype(np.float64)
            engulfing[0] = 0.0

            prev_body = np.abs(prev_close - prev_open)
            harami_bull = (
                (closes > opens)
                & (prev_close < prev_open)
                & (opens >= np.minimum(prev_open, prev_close))
                & (closes <= np.maximum(prev_open, prev_close))
                & (prev_body > body * 1.15)
            )
            harami_bear = (
                (closes < opens)
                & (prev_close > prev_open)
                & (opens <= np.maximum(prev_open, prev_close))
                & (closes >= np.minimum(prev_open, prev_close))
                & (prev_body > body * 1.15)
            )
            harami = harami_bull.astype(np.float64) - harami_bear.astype(np.float64)
            harami[0] = 0.0

            midpoint_prev = (prev_open + prev_close) * 0.5
            piercing = (
                (prev_close < prev_open)
                & (closes > opens)
                & (opens < prev_low)
                & (closes > midpoint_prev)
                & (closes < prev_open)
            ).astype(np.float64)
            dark_cloud_cover = -(
                (prev_close > prev_open)
                & (closes < opens)
                & (opens > prev_high)
                & (closes < midpoint_prev)
                & (closes > prev_open)
            ).astype(np.float64)
            piercing[0] = 0.0
            dark_cloud_cover[0] = 0.0

            morning_star = np.zeros_like(closes, dtype=np.float64)
            evening_star = np.zeros_like(closes, dtype=np.float64)
            for idx in range(2, closes.size):
                first_bearish = closes[idx - 2] < opens[idx - 2]
                first_bullish = closes[idx - 2] > opens[idx - 2]
                small_middle = abs(closes[idx - 1] - opens[idx - 1]) <= 0.35 * (highs[idx - 1] - lows[idx - 1] + 1e-8)
                strong_bull = closes[idx] > opens[idx] and closes[idx] >= (opens[idx - 2] + closes[idx - 2]) / 2.0
                strong_bear = closes[idx] < opens[idx] and closes[idx] <= (opens[idx - 2] + closes[idx - 2]) / 2.0

                if first_bearish and small_middle and strong_bull:
                    morning_star[idx] = 1.0
                if first_bullish and small_middle and strong_bear:
                    evening_star[idx] = -1.0

            hanging_man = np.zeros_like(closes, dtype=np.float64)
            three_white_soldiers = np.zeros_like(closes, dtype=np.float64)
            three_black_crows = np.zeros_like(closes, dtype=np.float64)
            for idx in range(3, closes.size):
                uptrend = closes[idx - 1] > closes[idx - 3]
                if uptrend and lower_shadow[idx] / candle_range[idx] > 0.55 and upper_shadow[idx] / candle_range[idx] < 0.2:
                    hanging_man[idx] = -1.0

                bullish_chain = closes[idx] > opens[idx] and closes[idx - 1] > opens[idx - 1] and closes[idx - 2] > opens[idx - 2]
                bearish_chain = closes[idx] < opens[idx] and closes[idx - 1] < opens[idx - 1] and closes[idx - 2] < opens[idx - 2]
                rising_closes = closes[idx] > closes[idx - 1] > closes[idx - 2]
                falling_closes = closes[idx] < closes[idx - 1] < closes[idx - 2]

                if bullish_chain and rising_closes:
                    three_white_soldiers[idx] = 1.0
                if bearish_chain and falling_closes:
                    three_black_crows[idx] = -1.0

            marubozu = np.where(
                (body / candle_range > 0.78) & (upper_shadow / candle_range < 0.08) & (lower_shadow / candle_range < 0.08),
                np.sign(closes - opens),
                0.0,
            )

            patterns = {
                "doji": np.tanh(doji),
                "engulfing": np.tanh(engulfing),
                "harami": np.tanh(harami),
                "piercing": np.tanh(piercing),
                "dark_cloud_cover": np.tanh(dark_cloud_cover),
                "hammer": np.tanh(hammer),
                "inverted_hammer": np.tanh(inverted_hammer),
                "shooting_star": np.tanh(shooting_star),
                "spinning_top": np.tanh(spinning_top),
                "dragonfly_doji": np.tanh(dragonfly_doji),
                "gravestone_doji": np.tanh(gravestone_doji),
                "morning_star": np.tanh(morning_star),
                "evening_star": np.tanh(evening_star),
                "hanging_man": np.tanh(hanging_man),
                "three_white_soldiers": np.tanh(three_white_soldiers),
                "three_black_crows": np.tanh(three_black_crows),
                "marubozu": np.tanh(marubozu),
            }

        # Blend a richer set of candlestick patterns into one compact signal.
        weights = {
            "engulfing": 0.16,
            "harami": 0.10,
            "piercing": 0.09,
            "dark_cloud_cover": 0.09,
            "hammer": 0.08,
            "inverted_hammer": 0.07,
            "shooting_star": 0.08,
            "morning_star": 0.08,
            "evening_star": 0.08,
            "hanging_man": 0.06,
            "three_white_soldiers": 0.08,
            "three_black_crows": 0.08,
            "marubozu": 0.06,
            "spinning_top": 0.05,
            "dragonfly_doji": 0.04,
            "gravestone_doji": 0.04,
            "doji": 0.03,
        }
        combined = np.zeros_like(closes, dtype=np.float64)
        for name, weight in weights.items():
            if name in patterns:
                combined += patterns[name] * weight

        return np.tanh(combined), patterns

    def _realized_volatility_series(self, closes: np.ndarray) -> dict[str, np.ndarray]:
        if closes.size == 0:
            return {}

        returns = np.zeros_like(closes, dtype=np.float64)
        if closes.size > 1:
            returns[1:] = np.diff(np.log(np.maximum(closes, 1e-8)))

        series = pd.Series(returns, dtype="float64")
        output: dict[str, np.ndarray] = {}
        for window in (3, 5, 8, 10, 14, 20, 30, 40, 60, 80, 120, 160):
            output[str(window)] = series.rolling(window=window, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64)
        return output

    def _build_context_series(
        self,
        request: PredictRequest,
        sentiment_score: float | None = None,
        external_sentiment_score: float | None = None,
        external_covariate_signal: np.ndarray | None = None,
        external_covariate_scale: float | None = None,
    ) -> dict[str, Any]:
        opens = np.asarray([float(item.open) for item in request.latest_candles], dtype=np.float64)
        highs = np.asarray([float(item.high) for item in request.latest_candles], dtype=np.float64)
        lows = np.asarray([float(item.low) for item in request.latest_candles], dtype=np.float64)
        closes = np.asarray([float(item.close) for item in request.latest_candles], dtype=np.float64)
        volumes = np.asarray([float(item.volume) for item in request.latest_candles], dtype=np.float64)
        returns = np.zeros_like(closes, dtype=np.float64)
        if closes.size > 1:
            returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-8)

        pattern_signal, pattern_signals = self._pattern_signal_series(request)
        volatility = self._realized_volatility_series(closes)

        context: list[float] = []
        volume_log = np.log1p(np.maximum(volumes, 0.0))
        volume_mean = pd.Series(volume_log).rolling(window=20, min_periods=2).mean().fillna(method="bfill").fillna(0.0).to_numpy(dtype=np.float64)
        volume_std = pd.Series(volume_log).rolling(window=20, min_periods=2).std().fillna(1.0).to_numpy(dtype=np.float64)
        volume_z = (volume_log - volume_mean) / np.maximum(volume_std, 1e-6)

        momentum_3 = (
            pd.Series(returns, dtype="float64")
            .rolling(window=3, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        momentum_8 = (
            pd.Series(returns, dtype="float64")
            .rolling(window=8, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        fast_ema = pd.Series(closes, dtype="float64").ewm(span=8, adjust=False).mean().to_numpy(dtype=np.float64)
        slow_ema = pd.Series(closes, dtype="float64").ewm(span=21, adjust=False).mean().to_numpy(dtype=np.float64)
        trend_strength = (fast_ema - slow_ema) / np.maximum(closes, 1e-8)

        candle_range = np.maximum(highs - lows, 1e-8)
        body_ratio = (closes - opens) / candle_range
        upper_shadow = highs - np.maximum(opens, closes)
        lower_shadow = np.minimum(opens, closes) - lows
        wick_imbalance = (upper_shadow - lower_shadow) / candle_range

        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        true_range = np.maximum.reduce((highs - lows, np.abs(highs - prev_close), np.abs(lows - prev_close)))
        atr_14 = (
            pd.Series(true_range, dtype="float64")
            .rolling(window=14, min_periods=2)
            .mean()
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        atr_norm = atr_14 / np.maximum(closes, 1e-8)

        vol_3 = volatility.get("3", np.zeros_like(closes, dtype=np.float64))
        vol_5 = volatility.get("5", np.zeros_like(closes, dtype=np.float64))
        vol_8 = volatility.get("8", np.zeros_like(closes, dtype=np.float64))
        vol_14 = volatility.get("14", np.zeros_like(closes, dtype=np.float64))
        vol_20 = volatility.get("20", np.zeros_like(closes, dtype=np.float64))
        vol_40 = volatility.get("40", np.zeros_like(closes, dtype=np.float64))
        vol_80 = volatility.get("80", np.zeros_like(closes, dtype=np.float64))
        vol_120 = volatility.get("120", np.zeros_like(closes, dtype=np.float64))
        vol_160 = volatility.get("160", np.zeros_like(closes, dtype=np.float64))

        vol_regime = (vol_8 - vol_20) / np.maximum(vol_80, 1e-6)
        vol_expansion = (vol_5 / np.maximum(vol_20, 1e-6)) - 1.0
        vol_persistence = (vol_20 - vol_80) / np.maximum(vol_160 + 1e-6, 1e-6)
        vol_acceleration = (vol_3 - vol_14) / np.maximum(vol_40 + 1e-6, 1e-6)
        vol_shock = (vol_14 - vol_120) / np.maximum(vol_160 + 1e-6, 1e-6)

        sentiment_feature = 0.0
        if sentiment_score is not None:
            sentiment_feature += max(-1.0, min(1.0, float(sentiment_score))) * 0.006
        if external_sentiment_score is not None:
            sentiment_feature += max(-1.0, min(1.0, float(external_sentiment_score))) * 0.024

        external_direction_boost = (
            np.tanh(max(-1.0, min(1.0, float(external_sentiment_score or 0.0))) * 1.35) * 0.010
        )

        covariate_scale = float(external_covariate_scale or 0.0)
        covariate_signal = external_covariate_signal
        if covariate_signal is None or covariate_signal.size == 0:
            covariate_signal = np.zeros_like(closes, dtype=np.float64)

        for index, close in enumerate(closes):
            pattern_feature = float(pattern_signal[index]) if index < pattern_signal.size else 0.0
            pattern_impulse = pattern_feature * (1.0 + abs(pattern_feature) * 0.85)
            covariate_adjustment = 0.0
            if covariate_scale > 0.0 and index < covariate_signal.size:
                covariate_adjustment = (
                    math.exp(np.clip(float(covariate_signal[index]), -3.0, 3.0) * covariate_scale) - 1.0
                )
            adjustment = (
                np.tanh(volume_z[index]) * 0.006
                + np.tanh(momentum_3[index] * 52.0) * 0.016
                + np.tanh(momentum_8[index] * 34.0) * 0.013
                + np.tanh(trend_strength[index] * 120.0) * 0.018
                + np.tanh(vol_regime[index] * 1.05) * 0.024
                + np.tanh(vol_expansion[index] * 1.30) * 0.017
                + np.tanh(vol_persistence[index] * 0.95) * 0.011
                + np.tanh(vol_acceleration[index] * 1.10) * 0.013
                + np.tanh(vol_shock[index] * 0.95) * 0.011
                + np.tanh(atr_norm[index] * 7.5) * 0.012
                + np.tanh(body_ratio[index] * 1.8) * 0.008
                + np.tanh(wick_imbalance[index] * 1.6) * 0.006
                + pattern_impulse * 0.035
                + sentiment_feature
                + external_direction_boost
                + covariate_adjustment
            )
            adjustment = float(max(-0.18, min(0.18, adjustment)))
            context.append(float(close * (1.0 + adjustment)))

        # Keep context compact for low-latency CPU inference in Fargate.
        return {
            "context": context[-512:],
            "closes": closes,
            "returns": returns,
            "pattern_signal": pattern_signal,
            "pattern_signals": pattern_signals,
            "volatility": volatility,
            "covariate_signal": covariate_signal,
        }

    def _build_multivariate_context(
        self,
        request: PredictRequest,
        sentiment_score: float | None = None,
        external_sentiment_score: float | None = None,
        external_covariate_signal: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build a (1, n_variates, T) tensor for Chronos-2 multivariate input.

        Variates:
          0: Close prices (primary target — only this variate is forecast)
          1: Log-volume z-score
          2: Candlestick pattern signal
          3: External covariate signal (funding rate, OI, FnG, etc.)
          4: Blended sentiment (market + external)
        """
        closes = np.asarray([float(c.close) for c in request.latest_candles], dtype=np.float64)
        volumes = np.asarray([float(c.volume) for c in request.latest_candles], dtype=np.float64)

        # Variate 1: Log-volume z-score
        vol_log = np.log1p(np.maximum(volumes, 0.0))
        vol_mean = pd.Series(vol_log).rolling(20, min_periods=2).mean().bfill().fillna(0.0).to_numpy(dtype=np.float64)
        vol_std = pd.Series(vol_log).rolling(20, min_periods=2).std().bfill().fillna(1.0).to_numpy(dtype=np.float64)
        vol_z = (vol_log - vol_mean) / np.maximum(vol_std, 1e-6)

        # Variate 2: Candlestick pattern signal
        pattern_signal, pattern_signals = self._pattern_signal_series(request)
        if pattern_signal.size < closes.size:
            pattern_signal = np.pad(pattern_signal, (closes.size - pattern_signal.size, 0), mode="constant")
        elif pattern_signal.size > closes.size:
            pattern_signal = pattern_signal[-closes.size:]

        # Variate 3: External covariate signal
        cov = external_covariate_signal
        if cov is None or cov.size != closes.size:
            cov = np.zeros_like(closes, dtype=np.float64)

        # Variate 4: Blended sentiment (constant across time, but provides info)
        sent_val = float(sentiment_score or 0.0)
        ext_sent_val = float(external_sentiment_score or 0.0)
        combined_sent = np.full_like(closes, sent_val * 0.3 + ext_sent_val * 0.7)

        # Stack variates, truncate to last 512 steps
        ctx_len = min(512, closes.size)
        variates = np.stack([
            closes[-ctx_len:],
            vol_z[-ctx_len:],
            pattern_signal[-ctx_len:],
            cov[-ctx_len:],
            combined_sent[-ctx_len:],
        ], axis=0)  # shape: (5, ctx_len)

        # Chronos-2 expects (n_series, n_variates, T)
        tensor = torch.tensor(variates, dtype=torch.float32).unsqueeze(0)  # (1, 5, ctx_len)

        volatility = self._realized_volatility_series(closes)
        metadata = {
            "closes": closes,
            "pattern_signals": pattern_signals,
            "volatility": volatility,
            "covariate_signal": cov,
        }
        return tensor, metadata

    def _estimate_sentiment_score(
        self,
        request: PredictRequest,
        *,
        force_external_refresh: bool,
        require_external: bool,
    ) -> tuple[float, str, float, str]:
        candles = request.latest_candles
        frame = pd.DataFrame(
            {
                "open": [float(item.open) for item in candles],
                "high": [float(item.high) for item in candles],
                "low": [float(item.low) for item in candles],
                "close": [float(item.close) for item in candles],
                "volume": [float(item.volume) for item in candles],
            }
        )

        if frame.empty:
            raise ValueError("Sentiment scoring requires non-empty candle data")

        try:
            score, source, external_score, external_source = self.sentiment_scorer.score_latest(
                request.symbol.upper(),
                frame,
                force_external_refresh=force_external_refresh,
                require_external=require_external,
            )
            return (
                max(-1.0, min(1.0, float(score))),
                source,
                max(-1.0, min(1.0, float(external_score))),
                external_source,
            )
        except Exception as exc:
            logger.exception("External sentiment scoring failed for %s", request.symbol.upper())
            raise RuntimeError(f"External sentiment scoring failed: {exc}") from exc

    def _estimate_external_sentiment_with_retries(
        self,
        request: PredictRequest,
        *,
        max_attempts: int = 3,
    ) -> tuple[float, str, float, str]:
        last_error: RuntimeError | None = None
        force_refresh = bool(self.settings.external_sentiment_force_refresh_per_request)
        require_external = bool(self.settings.external_sentiment_require_live_sources)

        for attempt in range(1, max_attempts + 1):
            try:
                return self._estimate_sentiment_score(
                    request,
                    force_external_refresh=force_refresh,
                    require_external=require_external,
                )
            except RuntimeError as sentiment_error:
                last_error = sentiment_error
                if attempt >= max_attempts:
                    break

                backoff_seconds = 0.75 * attempt
                logger.warning(
                    "External sentiment attempt %s/%s failed for %s. Retrying in %.2fs: %s",
                    attempt,
                    max_attempts,
                    request.symbol.upper(),
                    backoff_seconds,
                    sentiment_error,
                )
                time.sleep(backoff_seconds)

        raise RuntimeError(
            f"External sentiment is required but unavailable after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _extract_numbers(self, text: str, horizon: int) -> list[float]:
        values = [float(item) for item in NUMERIC_REGEX.findall(text)]
        if not values:
            raise ValueError("Model text output did not include numeric forecast values")

        if len(values) < horizon:
            last = values[-1]
            values.extend([last] * (horizon - len(values)))

        return values[:horizon]

    def _as_sample_matrix(self, output: Any, horizon: int) -> np.ndarray:
        if isinstance(output, dict):
            for key in ("samples", "predictions", "forecast", "values"):
                if key in output:
                    return self._as_sample_matrix(output[key], horizon)

        if isinstance(output, (list, tuple)):
            array = np.asarray(output, dtype=np.float32)
        elif isinstance(output, torch.Tensor):
            array = output.detach().cpu().numpy().astype(np.float32)
        elif isinstance(output, np.ndarray):
            array = output.astype(np.float32)
        elif isinstance(output, str):
            return np.asarray([self._extract_numbers(output, horizon)], dtype=np.float32)
        else:
            raise TypeError(f"Unsupported model output type: {type(output)}")

        if array.ndim == 0:
            return np.full((1, horizon), float(array), dtype=np.float32)

        if array.ndim == 1:
            values = array[:horizon]
            if values.shape[0] < horizon:
                values = np.pad(values, (0, horizon - values.shape[0]), mode="edge")
            return values.reshape(1, horizon)

        if array.ndim == 2:
            if array.shape[1] == horizon:
                return array
            if array.shape[0] == horizon:
                return array.T

            trimmed = array[:, :horizon]
            if trimmed.shape[1] < horizon:
                pad_width = horizon - trimmed.shape[1]
                trimmed = np.pad(trimmed, ((0, 0), (0, pad_width)), mode="edge")
            return trimmed

        if array.ndim >= 3:
            # Typical shape from probabilistic forecast models: [batch, samples, horizon]
            candidate = array[0]
            if candidate.ndim == 2:
                return self._as_sample_matrix(candidate, horizon)

            flattened = candidate.reshape(-1)
            return self._as_sample_matrix(flattened, horizon)

        raise ValueError("Unable to convert model output into sample matrix")

    def _run_hf_quantile_forecast(
        self,
        loaded_model: LoadedForecastModel,
        context: list[float],
        horizon: int,
        quantile_levels: list[float],
    ) -> dict[float, np.ndarray] | None:
        model = loaded_model.model

        predict_quantiles = getattr(model, "predict_quantiles", None)
        if predict_quantiles is None:
            return None

        context_tensor_2d = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        context_tensor_3d = context_tensor_2d.unsqueeze(1)

        try:
            quantiles_result, _mean_result = predict_quantiles(
                inputs=context_tensor_3d,
                prediction_length=horizon,
                quantile_levels=quantile_levels,
            )
        except TypeError:
            try:
                quantiles_result, _mean_result = predict_quantiles(
                    context=context_tensor_3d,
                    horizon=horizon,
                    quantile_levels=quantile_levels,
                )
            except Exception:
                return None
        except Exception:
            return None

        if not isinstance(quantiles_result, (list, tuple)) or not quantiles_result:
            return None

        first = quantiles_result[0]
        if not isinstance(first, torch.Tensor):
            first = torch.as_tensor(first)

        # Chronos-2 returns (n_variates, horizon, n_quantiles). Take the first variate.
        if first.ndim == 2:
            horizon_by_quantile = first.detach().cpu().numpy()
        else:
            horizon_by_quantile = first[0].detach().cpu().numpy()

        if horizon_by_quantile.shape[0] != horizon:
            horizon_by_quantile = np.asarray(horizon_by_quantile, dtype=np.float32)
            horizon_by_quantile = horizon_by_quantile[:horizon]

        if horizon_by_quantile.shape[-1] != len(quantile_levels):
            return None

        quantile_values: dict[float, np.ndarray] = {}
        for idx, quantile in enumerate(quantile_levels):
            quantile_values[float(quantile)] = np.asarray(horizon_by_quantile[:, idx], dtype=np.float32)

        return quantile_values

    def _run_hf_inference(
        self,
        loaded_model: LoadedForecastModel,
        context: list[float],
        horizon: int,
        num_samples: int,
    ) -> np.ndarray:
        model = loaded_model.model
        context_tensor_2d = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        # Chronos-2 expects (n_series, n_variates, history_length) for tensor/ndarray inputs.
        context_tensor_3d = context_tensor_2d.unsqueeze(1)

        with torch.inference_mode():
            if hasattr(model, "predict"):
                for kwargs in (
                    {"inputs": context_tensor_3d, "prediction_length": horizon, "num_samples": num_samples},
                    {"inputs": context_tensor_3d, "prediction_length": horizon},
                    {"inputs": context_tensor_2d, "prediction_length": horizon, "num_samples": num_samples},
                    {"inputs": context_tensor_2d, "prediction_length": horizon},
                    {"inputs": context, "prediction_length": horizon, "num_samples": num_samples},
                    {"inputs": context, "prediction_length": horizon},
                    {"context": context_tensor_3d, "prediction_length": horizon, "num_samples": num_samples},
                    {"past_values": context_tensor_3d, "prediction_length": horizon, "num_samples": num_samples},
                    {"context": context_tensor_2d, "prediction_length": horizon, "num_samples": num_samples},
                    {"past_values": context_tensor_2d, "prediction_length": horizon, "num_samples": num_samples},
                    {"context": context, "prediction_length": horizon, "num_samples": num_samples},
                    {"past_values": context, "prediction_length": horizon, "num_samples": num_samples},
                ):
                    try:
                        output = model.predict(**kwargs)
                        return self._as_sample_matrix(output, horizon)
                    except (TypeError, ValueError):
                        continue

            if hasattr(model, "forecast"):
                for kwargs in (
                    {"context": context_tensor_3d, "horizon": horizon, "num_samples": num_samples},
                    {"context": context_tensor_2d, "horizon": horizon, "num_samples": num_samples},
                    {"context": context, "horizon": horizon, "num_samples": num_samples},
                ):
                    try:
                        output = model.forecast(**kwargs)
                        return self._as_sample_matrix(output, horizon)
                    except (TypeError, ValueError):
                        continue

            tokenizer = loaded_model.tokenizer
            if tokenizer is None:
                raise RuntimeError("Loaded model does not expose predict/forecast and tokenizer is unavailable")
            if not hasattr(model, "generate"):
                raise RuntimeError("Loaded model does not expose predict/forecast/generate methods")

            prompt = (
                "You are a time-series forecasting model. "
                f"Predict the next {horizon} close prices from this sequence: "
                + ",".join(f"{value:.8f}" for value in context[-240:])
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max(32, horizon * 10),
            )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)

            return self._as_sample_matrix(text, horizon)

    def _recent_step_volatility(self, closes: np.ndarray, window: int) -> float:
        if closes.size < 3:
            return 0.0

        log_returns = np.diff(np.log(np.maximum(closes, 1e-8)))
        if log_returns.size < 2:
            return 0.0

        scoped = log_returns[-window:] if log_returns.size > window else log_returns
        if scoped.size < 2:
            return 0.0

        return float(np.std(scoped))

    def _postprocess_quantile_variance(
        self,
        quantile_values: dict[float, np.ndarray],
        closes: np.ndarray,
        sentiment_score: float,
        external_sentiment_score: float,
    ) -> dict[float, np.ndarray]:
        """Post-processing with Dynamic Minimum Spread.

        No artificial variance scaling, wave/drift injection, or jitter.
        Applies only a realized-volatility floor: if the model's P5-P95
        spread is narrower than 0.5 * realized_vol of the last 50 candles,
        bands are symmetrically expanded to that floor.  If the model
        predicts wider spread, it is kept untouched.
        """
        if 0.5 not in quantile_values:
            return quantile_values

        median = np.asarray(quantile_values[0.5], dtype=np.float64)
        if median.size == 0 or closes.size < 3:
            return quantile_values

        # --- Step 1: Enforce monotonic quantile ordering ---
        ordered_quantiles = sorted(quantile_values.keys())
        stack = np.vstack([
            np.asarray(quantile_values[q], dtype=np.float64)
            for q in ordered_quantiles
        ])
        stack = np.sort(stack, axis=0)
        for idx, q in enumerate(ordered_quantiles):
            quantile_values[q] = stack[idx]

        # --- Step 2: Dynamic Minimum Spread based on realized volatility ---
        # Compute realized std-dev of log-returns over last 50 candles.
        rv_window = min(50, int(closes.size))
        rv_closes = closes[-rv_window:]
        if rv_closes.size >= 3:
            log_returns = np.diff(np.log(np.maximum(rv_closes, 1e-8)))
            realized_vol = float(np.std(log_returns))
        else:
            realized_vol = 0.0

        # Minimum per-step spread (in price) = 0.5 * realized_vol * anchor_price.
        # This ensures bands reflect at least half the recent market volatility.
        min_spread_ratio = 0.5 * max(realized_vol, 0.0)
        if min_spread_ratio <= 0.0:
            return quantile_values

        # Identify the lowest and highest quantile levels.
        q_low = ordered_quantiles[0]   # e.g., 0.1
        q_high = ordered_quantiles[-1]  # e.g., 0.9
        if q_low == q_high:
            return quantile_values

        lower_band = np.asarray(quantile_values[q_low], dtype=np.float64)
        upper_band = np.asarray(quantile_values[q_high], dtype=np.float64)

        for step_idx in range(median.size):
            anchor = max(1e-8, float(median[step_idx]))
            # Scale floor with sqrt(step) for time-expanding uncertainty.
            step_floor = anchor * min_spread_ratio * math.sqrt(step_idx + 1)
            current_spread = float(upper_band[step_idx] - lower_band[step_idx])

            if current_spread >= step_floor:
                continue  # Model spread is wide enough — no intervention.

            # Symmetric expansion around median to reach the floor.
            half_deficit = (step_floor - current_spread) / 2.0
            lower_band[step_idx] = max(1e-8, lower_band[step_idx] - half_deficit)
            upper_band[step_idx] = upper_band[step_idx] + half_deficit

        quantile_values[q_low] = lower_band
        quantile_values[q_high] = upper_band

        # Re-sort to guarantee ordering after expansion.
        stack = np.vstack([quantile_values[q] for q in ordered_quantiles])
        stack = np.sort(stack, axis=0)
        for idx, q in enumerate(ordered_quantiles):
            quantile_values[q] = stack[idx]

        return quantile_values

    def _build_pattern_markers(
        self,
        request: PredictRequest,
        pattern_signals: dict[str, np.ndarray],
    ) -> list[PatternMarker]:
        if not pattern_signals:
            return []

        markers: list[PatternMarker] = []
        candle_count = len(request.latest_candles)
        start_index = max(0, candle_count - 220)

        for index in range(start_index, candle_count):
            strongest_name = ""
            strongest_value = 0.0

            for name, values in pattern_signals.items():
                if index >= values.size:
                    continue
                value = float(values[index])
                if abs(value) < 0.45:
                    continue
                if abs(value) > abs(strongest_value):
                    strongest_name = name
                    strongest_value = value

            if not strongest_name:
                continue

            if strongest_value > 0.12:
                direction = "bullish"
            elif strongest_value < -0.12:
                direction = "bearish"
            else:
                direction = "neutral"

            markers.append(
                PatternMarker(
                    timestamp=request.latest_candles[index].timestamp,
                    pattern=strongest_name,
                    direction=direction,
                    strength=round(strongest_value, 4),
                )
            )

        return markers[-40:]

    def _build_volatility_bands(
        self,
        median_forecast: np.ndarray,
        volatility: dict[str, np.ndarray],
    ) -> list[VolatilityBand]:
        if median_forecast.size == 0 or not volatility:
            return []

        bands: list[VolatilityBand] = []
        for label, key in (("rv20", "20"), ("rv40", "40")):
            vol_series = volatility.get(key)
            if vol_series is None or vol_series.size == 0:
                continue

            base_vol = float(vol_series[-1])
            if not math.isfinite(base_vol):
                continue
            base_vol = float(np.clip(base_vol, 0.0004, 0.0800))

            lower: list[float] = []
            upper: list[float] = []
            for step, value in enumerate(median_forecast, start=1):
                price = float(max(1e-8, value))
                width = max(price * base_vol * math.sqrt(step), price * 0.0008)
                lower.append(round(max(1e-8, price - width), 8))
                upper.append(round(price + width, 8))

            bands.append(VolatilityBand(label=label, lower=lower, upper=upper))

        return bands

    def predict(self, request: PredictRequest) -> PredictResponse:
        closes = [float(candle.close) for candle in request.latest_candles]
        if len(closes) < 20:
            raise ValueError("At least 20 candles are required for robust model inference")

        last_price = closes[-1]
        horizon = int(request.horizon)
        num_samples = max(10, min(64, int(self.settings.inference_num_samples)))

        sentiment_score, sentiment_source, external_sentiment_score, external_sentiment_source = (
            self._estimate_external_sentiment_with_retries(request)
        )

        sentiment_snapshot = self.sentiment_scorer.get_external_feature_snapshot(
            request.symbol.upper(),
            force_external_refresh=False,
        )
        try:
            covariate_signal, covariate_latest = build_external_covariate_signal(
                candles=request.latest_candles,
                symbol=request.symbol,
                timeframe=request.timeframe,
                sentiment_snapshot={
                    "fear_greed_index": sentiment_snapshot.fear_greed_index,
                    "crypto_news_sentiment": sentiment_snapshot.crypto_news_sentiment,
                    "x_sentiment_score": sentiment_snapshot.x_sentiment_score,
                    "event_impact_score": sentiment_snapshot.event_impact_score,
                },
                settings=self.settings,
            )
        except Exception as exc:
            logger.warning("External covariate build failed: %s", exc)
            covariate_signal = np.asarray([], dtype=np.float64)
            covariate_latest = {}

        feature_bundle = self._build_context_series(
            request,
            sentiment_score=sentiment_score,
            external_sentiment_score=external_sentiment_score,
            external_covariate_signal=covariate_signal,
            external_covariate_scale=self.settings.external_covariate_scale,
        )
        feature_bundle["external_covariates"] = covariate_latest
        context_series = feature_bundle["context"]

        # Build multivariate tensor for Chronos-2 (5 variates: close, vol_z, pattern, covariate, sentiment).
        # Stored in feature_bundle for downstream use; falls back to single-variate context_series.
        try:
            mv_tensor, mv_metadata = self._build_multivariate_context(
                request,
                sentiment_score=sentiment_score,
                external_sentiment_score=external_sentiment_score,
                external_covariate_signal=covariate_signal,
            )
            feature_bundle["multivariate_tensor"] = mv_tensor
            feature_bundle["pattern_signals"] = mv_metadata.get("pattern_signals", feature_bundle.get("pattern_signals", {}))
            feature_bundle["volatility"] = mv_metadata.get("volatility", feature_bundle.get("volatility", {}))
        except Exception as exc:
            logger.warning("Multivariate context build failed, using single-variate: %s", exc)
            feature_bundle["multivariate_tensor"] = None

        try:
            loaded_model = self._load_model()
            logger.info(
                "Model loaded for prediction: name=%s version=%s source=%s effective=%s",
                loaded_model.model_name,
                loaded_model.model_version,
                loaded_model.requested_source,
                loaded_model.effective_source,
            )
            quantiles = sorted({float(max(0.0, min(1.0, value))) for value in request.quantiles})
            if 0.5 not in quantiles:
                quantiles.append(0.5)
                quantiles = sorted(quantiles)

            quantile_values = self._run_hf_quantile_forecast(
                loaded_model=loaded_model,
                context=context_series,
                horizon=horizon,
                quantile_levels=quantiles,
            )
            forecast_detail = f"quantiles={len(quantiles)}"

            if quantile_values is None:
                sample_matrix = self._run_hf_inference(
                    loaded_model=loaded_model,
                    context=context_series,
                    horizon=horizon,
                    num_samples=num_samples,
                )

                if sample_matrix.size == 0:
                    raise RuntimeError("Chronos model returned empty forecast samples")

                if sample_matrix.shape[1] != horizon:
                    sample_matrix = self._as_sample_matrix(sample_matrix, horizon)

                sample_matrix = self._apply_sentiment(sample_matrix, sentiment_score)

                quantile_values = {quantile: np.quantile(sample_matrix, quantile, axis=0) for quantile in quantiles}
                forecast_detail = f"samples={sample_matrix.shape[0]}"
            else:
                adjustment = 1.0
                if sentiment_score is not None:
                    bounded = max(-1.0, min(1.0, sentiment_score))
                    adjustment = 1.0 + bounded * 0.002
                if adjustment != 1.0:
                    for quantile in list(quantile_values.keys()):
                        quantile_values[quantile] = quantile_values[quantile] * adjustment

            quantile_values = self._postprocess_quantile_variance(
                quantile_values,
                closes=np.asarray(feature_bundle["closes"], dtype=np.float64),
                sentiment_score=sentiment_score,
                external_sentiment_score=external_sentiment_score,
            )

            model_name = loaded_model.model_name
            model_version = loaded_model.model_version

            # Log feature completeness
            non_zero_covariates = sum(1 for v in covariate_latest.values() if v != 0.0) if covariate_latest else 0
            total_covariates = len(covariate_latest) if covariate_latest else 0
            logger.info(
                "Feature completeness for %s: context_len=%d patterns=%d vol_windows=%d covariates=%d/%d",
                request.symbol.upper(),
                len(context_series),
                len(feature_bundle.get("pattern_signals", {})),
                len(feature_bundle.get("volatility", {})),
                non_zero_covariates,
                total_covariates,
            )
        except Exception as exc:
            logger.exception("Chronos inference failed for model_s3_uri=%s", self.settings.model_s3_uri)
            raise RuntimeError(f"Chronos model load/inference failed: {exc}") from exc

        median_forecast = quantile_values[0.5]
        prediction_array = [round(float(value), 8) for value in median_forecast.tolist()]

        lower_q = min(quantiles)
        upper_q = max(quantiles)
        lower_series = [round(float(value), 8) for value in quantile_values[lower_q].tolist()]
        upper_series = [round(float(value), 8) for value in quantile_values[upper_q].tolist()]

        first_interval = ConfidenceInterval(
            lower=lower_series[0],
            upper=upper_series[0],
        )

        interval_width = max(1e-8, first_interval.upper - first_interval.lower)
        confidence = max(0.05, min(0.99, 1.0 - (interval_width / max(last_price, 1e-8))))

        horizon_target = prediction_array[-1]
        delta = horizon_target - last_price
        threshold = max(1e-8, abs(last_price) * 0.0005)
        if abs(delta) <= threshold:
            trend_direction = "flat"
        else:
            trend_direction = "up" if delta > 0 else "down"

        returns = []
        for index in range(1, len(closes)):
            previous = closes[index - 1]
            if previous > 0:
                returns.append((closes[index] - previous) / previous)
        volatility = pstdev(returns) if len(returns) > 1 else 0.0

        confidence_bands = [
            ConfidenceBand(
                quantile=quantile,
                values=[round(float(value), 8) for value in quantile_values[quantile].tolist()],
            )
            for quantile in quantiles
        ]
        pattern_markers = self._build_pattern_markers(
            request,
            pattern_signals=feature_bundle.get("pattern_signals", {}),
        )
        volatility_bands = self._build_volatility_bands(
            median_forecast=median_forecast,
            volatility=feature_bundle.get("volatility", {}),
        )

        covariate_latest = feature_bundle.get("external_covariates", {}) or {}
        covariate_detail = ""
        if covariate_latest:
            covariate_detail = (
                " covariates="
                f"fng={covariate_latest.get('fear_greed_index', 0.0):.1f} "
                f"news={covariate_latest.get('crypto_news_sentiment', 0.0):.3f} "
                f"x={covariate_latest.get('x_sentiment_score', 0.0):.3f} "
                f"funding={covariate_latest.get('funding_rate', 0.0):.5f} "
                f"oi={covariate_latest.get('open_interest', 0.0):.2f} "
                f"ls={covariate_latest.get('long_short_ratio', 0.0):.3f} "
                f"top_ls={covariate_latest.get('top_trader_long_short_ratio', 0.0):.3f} "
                f"taker={covariate_latest.get('taker_buy_sell_ratio', 0.0):.3f} "
                f"btc_dom={covariate_latest.get('btc_dominance', 0.0):.2f} "
                f"dxy={covariate_latest.get('macro_dxy', 0.0):.2f} "
                f"us10y={covariate_latest.get('macro_us10y', 0.0):.2f} "
                f"event={covariate_latest.get('event_impact_score', 0.0):.3f}"
            )

        explanation = (
            "Forecast blends Chronos inference with live candlestick patterns, "
            "multi-window realized volatility, and external covariates. "
            f"Horizon={horizon}, {forecast_detail}, volatility={volatility:.6f}, "
            f"sentiment={sentiment_score:.3f} ({sentiment_source}), "
            f"external={external_sentiment_score:.3f} ({external_sentiment_source})."
            f"{covariate_detail}"
        )

        return PredictResponse(
            symbol=request.symbol.upper(),
            timeframe=request.timeframe,
            horizon=horizon,
            predicted_price=prediction_array[0],
            prediction_array=prediction_array,
            confidence=round(float(confidence), 4),
            confidence_interval=first_interval,
            confidence_bands=confidence_bands,
            volatility_bands=volatility_bands,
            pattern_markers=pattern_markers,
            sentiment_score=round(float(sentiment_score), 4),
            sentiment_source=sentiment_source,
            external_sentiment_score=round(float(external_sentiment_score), 4),
            external_sentiment_source=external_sentiment_source,
            trend_direction=trend_direction,
            model_name=model_name,
            model_version=model_version,
            explanation=explanation,
            generated_at=datetime.now(timezone.utc),
        )


@lru_cache
def get_inference_service() -> ForecastInferenceService:
    return ForecastInferenceService()
