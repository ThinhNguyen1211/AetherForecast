from datetime import datetime, timezone
from functools import lru_cache
import logging
import math
import re
from statistics import pstdev
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.core.config import get_settings
from src.data.sentiment import SentimentScorer
from src.ml.model_loader import LoadedForecastModel, get_loaded_forecasting_model
from src.ml.schemas import ConfidenceBand, ConfidenceInterval, PredictRequest, PredictResponse

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
            x_sentiment_endpoint=self.settings.external_x_sentiment_endpoint,
            geopolitical_sentiment_endpoint=self.settings.external_geopolitical_sentiment_endpoint,
        )

    def _load_model(self) -> LoadedForecastModel:
        return get_loaded_forecasting_model(
            model_s3_uri=self.settings.model_s3_uri,
            fallback_model_id=self.settings.hf_model_fallback_id,
            tokenizer_fallback_id=self.settings.hf_tokenizer_fallback_id,
            model_cache_dir=self.settings.model_cache_dir,
            aws_region=self.settings.aws_region,
            endpoint_url=self.settings.aws_endpoint_url,
            torch_dtype=self.settings.model_torch_dtype,
            require_s3_model=self.settings.require_s3_model,
        )

    def _apply_sentiment(self, forecast: np.ndarray, sentiment_score: float | None) -> np.ndarray:
        if sentiment_score is None:
            return forecast

        bounded = max(-1.0, min(1.0, sentiment_score))
        adjustment = 1.0 + bounded * 0.002
        return forecast * adjustment

    def _pattern_signal_series(self, request: PredictRequest) -> np.ndarray:
        opens = np.asarray([float(item.open) for item in request.latest_candles], dtype=np.float64)
        highs = np.asarray([float(item.high) for item in request.latest_candles], dtype=np.float64)
        lows = np.asarray([float(item.low) for item in request.latest_candles], dtype=np.float64)
        closes = np.asarray([float(item.close) for item in request.latest_candles], dtype=np.float64)

        if closes.size == 0:
            return np.asarray([], dtype=np.float64)

        if talib is not None:
            try:
                doji = talib.CDLDOJI(opens, highs, lows, closes)
                engulfing = talib.CDLENGULFING(opens, highs, lows, closes)
                hammer = talib.CDLHAMMER(opens, highs, lows, closes)
                # Normalize strong candlestick events to a compact range.
                combined = doji * 0.3 + engulfing * 0.5 + hammer * 0.2
                return np.tanh(combined / 100.0)
            except Exception as exc:
                logger.warning("TA-Lib pattern extraction failed in inference: %s", exc)

        candle_range = np.maximum(highs - lows, 1e-8)
        body = np.abs(closes - opens)
        lower_shadow = np.minimum(opens, closes) - lows
        doji = (body / candle_range < 0.1).astype(np.float64)
        hammer = (lower_shadow / candle_range > 0.55).astype(np.float64)
        engulfing = np.sign(closes - opens).astype(np.float64)
        return np.tanh(doji * 0.2 + hammer * 0.3 + engulfing * 0.5)

    def _build_context_series(
        self,
        request: PredictRequest,
        sentiment_score: float | None = None,
        external_sentiment_score: float | None = None,
    ) -> list[float]:
        closes = np.asarray([float(item.close) for item in request.latest_candles], dtype=np.float64)
        returns = np.diff(closes) / np.maximum(closes[:-1], 1e-8) if closes.size > 1 else np.asarray([])
        pattern_signal = self._pattern_signal_series(request)

        context: list[float] = []

        for index, candle in enumerate(request.latest_candles):
            typical_price = (candle.open + candle.high + candle.low + candle.close) / 4.0
            volume_feature = math.log1p(max(candle.volume, 0.0)) * 0.0001

            # Include volatility and TA-Lib candlestick pattern signals in model context.
            start = max(0, index - 20)
            window_returns = returns[start:index] if returns.size > 0 else np.asarray([])
            volatility = float(np.std(window_returns)) if window_returns.size > 1 else 0.0
            volatility_feature = max(-0.002, min(0.002, volatility * 0.05))

            pattern_feature = 0.0
            if index < pattern_signal.size:
                pattern_feature = float(pattern_signal[index]) * 0.001

            sentiment_feature = 0.0
            if sentiment_score is not None:
                sentiment_feature += max(-1.0, min(1.0, float(sentiment_score))) * 0.0008
            if external_sentiment_score is not None:
                sentiment_feature += max(-1.0, min(1.0, float(external_sentiment_score))) * 0.0012

            context.append(
                float(
                    typical_price
                    * (1.0 + volume_feature + volatility_feature + pattern_feature + sentiment_feature)
                )
            )

        # Keep context compact for low-latency CPU inference in Fargate.
        return context[-512:]

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

    def predict(self, request: PredictRequest) -> PredictResponse:
        closes = [float(candle.close) for candle in request.latest_candles]
        if len(closes) < 20:
            raise ValueError("At least 20 candles are required for robust model inference")

        last_price = closes[-1]
        horizon = int(request.horizon)
        num_samples = max(10, min(64, int(self.settings.inference_num_samples)))

        # Enforce live, backend-computed sentiment for every prediction.
        sentiment_score, sentiment_source, external_sentiment_score, external_sentiment_source = (
            self._estimate_sentiment_score(
                request,
                force_external_refresh=True,
                require_external=True,
            )
        )

        context_series = self._build_context_series(
            request,
            sentiment_score=sentiment_score,
            external_sentiment_score=external_sentiment_score,
        )

        try:
            loaded_model = self._load_model()
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

            model_name = loaded_model.model_name
            model_version = loaded_model.model_version
            model_requested_source = loaded_model.requested_source
            model_effective_source = loaded_model.effective_source
            model_torch_dtype = loaded_model.torch_dtype
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

        requested = (model_requested_source or "").rstrip("/")
        effective = (model_effective_source or "").rstrip("/")
        if requested and effective and requested != effective:
            model_origin = f"requested={requested}, promoted={effective}"
        else:
            model_origin = f"source={effective or requested}"

        model_runtime = f"Hugging Face Chronos model {model_name}:{model_version} (dtype={model_torch_dtype}, {model_origin})"
        explanation = (
            f"Forecast generated by {model_runtime}. "
            f"Horizon={horizon}, {forecast_detail}, volatility={volatility:.6f}, "
            f"sentiment={sentiment_score:.3f} ({sentiment_source}), "
            f"external={external_sentiment_score:.3f} ({external_sentiment_source})."
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
