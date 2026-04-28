from __future__ import annotations

from dataclasses import dataclass
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
import re
import threading
import xml.etree.ElementTree as et
from urllib.parse import quote_plus

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


_POSITIVE_KEYWORDS = {
    "bullish",
    "surge",
    "rally",
    "etf",
    "approval",
    "adoption",
    "breakout",
    "record high",
    "optimism",
}
_NEGATIVE_KEYWORDS = {
    "bearish",
    "crash",
    "selloff",
    "hack",
    "ban",
    "war",
    "conflict",
    "sanction",
    "recession",
    "inflation shock",
}

_DEFAULT_GEOPOLITICAL_RSS_URLS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]

_DEFAULT_CRYPTO_NEWS_RSS_URLS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]

_DEFAULT_X_RSS_SOURCES = [
    "https://nitter.net/search/rss?f=tweets&q={query}",
    "https://nitter.poast.org/search/rss?f=tweets&q={query}",
]

_DEFAULT_EVENT_KEYWORDS = [
    "elon",
    "trump",
    "halving",
    "etf",
    "sec",
    "fed",
    "rate cut",
    "rate hike",
    "cpi",
    "inflation",
    "blackrock",
    "whale",
    "regulation",
    "crash",
    "pump",
    "dump",
    "hack",
    "exploit",
    "defi",
    "stablecoin",
    "cbdc",
    "binance",
    "coinbase",
    "grayscale",
    "microstrategy",
]

_DEFAULT_X_TOPICS = [
    "bitcoin",
    "crypto market",
    "ethereum",
    "solana",
    "etf",
    "halving",
    "elon",
    "trump",
    "binance",
    "whale",
    "regulation",
    "trump crypto",
    "stablecoin",
    "tether",
]


@dataclass(frozen=True)
class ExternalSentimentSnapshot:
    score: float
    source: str
    fear_greed_index: float
    crypto_news_sentiment: float
    x_sentiment_score: float
    geopolitical_sentiment: float
    event_impact_score: float


class SentimentScorer:
    def __init__(
        self,
        mode: str = "simple",
        model_id: str = "ProsusAI/finbert",
        cache_dir: str = "/tmp/aetherforecast-sentiment-cache",
        external_enabled: bool = True,
        external_refresh_seconds: int = 900,
        news_rss_urls: list[str] | None = None,
        news_api_endpoint: str | None = None,
        news_api_key: str | None = None,
        news_api_query: str | None = None,
        news_api_max_items: int = 60,
        x_sentiment_endpoint: str | None = None,
        x_search_endpoint: str | None = None,
        x_search_bearer_token: str | None = None,
        x_search_query: str | None = None,
        x_search_max_items: int = 60,
        geopolitical_sentiment_endpoint: str | None = None,
        event_keywords: list[str] | None = None,
    ) -> None:
        self.mode = mode.lower().strip()
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._classifier = None
        self._classifier_lock = threading.Lock()
        self._classifier_load_attempted = False
        self.external_enabled = external_enabled
        self.external_refresh_seconds = max(60, int(external_refresh_seconds))
        self.news_rss_urls = news_rss_urls or []
        self.news_api_endpoint = news_api_endpoint
        self.news_api_key = news_api_key
        self.news_api_query = (news_api_query or "").strip()
        self.news_api_max_items = max(10, int(news_api_max_items))
        self.x_sentiment_endpoint = x_sentiment_endpoint
        self.x_search_endpoint = x_search_endpoint
        self.x_search_bearer_token = x_search_bearer_token
        self.x_search_query = (x_search_query or "").strip()
        self.x_search_max_items = max(10, int(x_search_max_items))
        self.geopolitical_sentiment_endpoint = geopolitical_sentiment_endpoint
        self.event_keywords = [item.strip().lower() for item in (event_keywords or _DEFAULT_EVENT_KEYWORDS) if item]
        self._external_cache: dict[str, tuple[ExternalSentimentSnapshot, datetime]] = {}
        self._external_cache_lock = threading.Lock()

        if self.mode == "hf" and pipeline is None:
            logger.warning("Transformers pipeline is unavailable; falling back to simple sentiment mode.")
            self.mode = "simple"

    def _ensure_classifier_loaded(self) -> None:
        if self.mode != "hf":
            return

        if self._classifier is not None:
            return

        if pipeline is None:
            self.mode = "simple"
            return

        with self._classifier_lock:
            if self._classifier is not None or self._classifier_load_attempted:
                return

            self._classifier_load_attempted = True

            try:
                self._classifier = pipeline(
                    "sentiment-analysis",
                    model=self.model_id,
                    tokenizer=self.model_id,
                    truncation=True,
                    device=-1,
                    model_kwargs={"low_cpu_mem_usage": True},
                )
                logger.info("Loaded HF sentiment pipeline: %s", self.model_id)
            except TypeError:
                # Older pipeline signature may not accept model_kwargs.
                try:
                    self._classifier = pipeline(
                        "sentiment-analysis",
                        model=self.model_id,
                        tokenizer=self.model_id,
                        truncation=True,
                        device=-1,
                    )
                    logger.info("Loaded HF sentiment pipeline: %s", self.model_id)
                except Exception as exc:
                    logger.warning(
                        "Unable to load HF sentiment model (%s). Falling back to simple mode.",
                        exc,
                    )
                    self._classifier = None
                    self.mode = "simple"
            except Exception as exc:
                logger.warning(
                    "Unable to load HF sentiment model (%s). Falling back to simple mode.",
                    exc,
                )
                self._classifier = None
                self.mode = "simple"

    def _fetch_json_score(self, url: str, score_keys: tuple[str, ...]) -> float | None:
        if not url:
            return None

        try:
            with httpx.Client(timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()

            for key in score_keys:
                if key in payload:
                    value = float(payload[key])
                    return max(-1.0, min(1.0, value))
        except Exception as exc:
            logger.warning("External sentiment endpoint failed (%s): %s", url, exc)

        return None

    def _fetch_fear_greed_index(self) -> float | None:
        url = "https://api.alternative.me/fng/?limit=1"
        try:
            with httpx.Client(timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()

            values = payload.get("data", [])
            if not values:
                return None

            value = float(values[0].get("value", 50.0))
            return max(0.0, min(100.0, value))
        except Exception as exc:
            logger.warning("Fear/Greed fetch failed: %s", exc)
            return None

    def _extract_headlines_from_payload(self, payload: object, max_items: int) -> list[str]:
        items: list[object] = []

        if isinstance(payload, dict):
            for key in ("data", "results", "articles", "posts"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    items = candidate
                    break
        elif isinstance(payload, list):
            items = payload

        headlines: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("title", "headline", "text", "summary", "description"):
                text = item.get(key)
                if text:
                    headlines.append(str(text))
                    break
            if len(headlines) >= max_items:
                break

        return headlines

    def _fetch_news_api_headlines(self, symbol: str, max_items: int) -> list[str]:
        if not self.news_api_endpoint:
            return []

        query = self.news_api_query or symbol
        params: dict[str, str | int] = {}
        headers: dict[str, str] = {}

        endpoint_lower = self.news_api_endpoint.lower()
        if self.news_api_key:
            if "cryptopanic" in endpoint_lower:
                params["auth_token"] = self.news_api_key
                params["kind"] = "news"
                params["filter"] = "hot"
            elif "newsapi" in endpoint_lower:
                params["apiKey"] = self.news_api_key
            elif "cryptonews" in endpoint_lower:
                params["token"] = self.news_api_key
            else:
                params["api_key"] = self.news_api_key

        if query:
            params.setdefault("q", query)
            params.setdefault("query", query)

        params.setdefault("limit", max_items)
        params.setdefault("pageSize", max_items)

        try:
            with httpx.Client(timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=True) as client:
                response = client.get(self.news_api_endpoint, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()

            return self._extract_headlines_from_payload(payload, max_items)
        except Exception as exc:
            logger.warning("News API fetch failed (%s): %s", self.news_api_endpoint, exc)
            return []

    def _fetch_x_search_headlines(self, query: str, max_items: int) -> list[str]:
        if not self.x_search_endpoint or not self.x_search_bearer_token:
            return []

        params = {"query": query, "max_results": min(100, max(10, int(max_items)))}
        headers = {"Authorization": f"Bearer {self.x_search_bearer_token}"}

        try:
            with httpx.Client(timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=True) as client:
                response = client.get(self.x_search_endpoint, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()

            items = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(items, list):
                return []

            headlines = []
            for item in items:
                if isinstance(item, dict) and item.get("text"):
                    headlines.append(str(item.get("text")))
                if len(headlines) >= max_items:
                    break
            return headlines
        except Exception as exc:
            logger.warning("X search fetch failed (%s): %s", self.x_search_endpoint, exc)
            return []

    def _event_impact_score(self, headlines: list[str]) -> float:
        if not headlines or not self.event_keywords:
            return 0.0

        hits = 0
        counted = 0
        for headline in headlines:
            text = re.sub(r"\s+", " ", headline.lower()).strip()
            if not text:
                continue
            counted += 1
            if any(keyword in text for keyword in self.event_keywords):
                hits += 1

        if counted == 0:
            return 0.0

        intensity = hits / max(1, counted)
        return float(np.tanh(intensity * 2.5))

    def _collect_rss_titles(self, urls: list[str], max_items: int) -> list[str]:
        titles: list[str] = []
        if not urls:
            return titles

        seen: set[str] = set()
        for url in urls:
            if len(titles) >= max_items:
                break

            try:
                with httpx.Client(timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    xml_text = response.text

                root = et.fromstring(xml_text)
                for item in root.findall(".//item/title"):
                    title = (item.text or "").strip()
                    if not title:
                        continue

                    normalized = re.sub(r"\s+", " ", title)
                    if normalized in seen:
                        continue

                    seen.add(normalized)
                    titles.append(normalized)
                    if len(titles) >= max_items:
                        break
            except Exception as exc:
                logger.warning("RSS fetch failed (%s): %s", url, exc)

        return titles

    def _collect_news_headlines(self, max_headlines: int = 60) -> list[str]:
        headlines: list[str] = []
        headlines.extend(self._collect_rss_titles(self.news_rss_urls, max_headlines))

        # Also fetch from default crypto news RSS if not enough headlines
        if len(headlines) < max_headlines:
            crypto_rss = self._collect_rss_titles(_DEFAULT_CRYPTO_NEWS_RSS_URLS, max_headlines - len(headlines))
            seen = {re.sub(r"\s+", " ", title).strip() for title in headlines}
            for headline in crypto_rss:
                normalized = re.sub(r"\s+", " ", headline).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                headlines.append(normalized)
                if len(headlines) >= max_headlines:
                    break

        if self.news_api_endpoint:
            api_headlines = self._fetch_news_api_headlines("crypto", max_headlines)
            seen = {re.sub(r"\s+", " ", title).strip() for title in headlines}
            for headline in api_headlines:
                normalized = re.sub(r"\s+", " ", headline).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                headlines.append(normalized)
                if len(headlines) >= max_headlines:
                    break

        return headlines[:max_headlines]

    def _collect_geopolitical_headlines(self, max_headlines: int = 20) -> list[str]:
        if self.geopolitical_sentiment_endpoint:
            return []
        return self._collect_rss_titles(_DEFAULT_GEOPOLITICAL_RSS_URLS, max_headlines)

    def _collect_x_headlines(self, symbol: str, max_items: int = 40) -> list[str]:
        if self.x_sentiment_endpoint:
            return []

        symbol_token = symbol.upper().replace("USDT", "").replace("USD", "")
        topics = [symbol_token or symbol.upper(), *_DEFAULT_X_TOPICS]
        if self.event_keywords:
            topics.extend(self.event_keywords[:6])

        headlines: list[str] = []
        x_query = self.x_search_query or " OR ".join(sorted(set(topics)))
        if x_query:
            x_query = f"{x_query} lang:en"
            headlines.extend(self._fetch_x_search_headlines(x_query, max_items))

        urls: list[str] = []
        for topic in topics:
            query = quote_plus(f"{topic} lang:en")
            for template in _DEFAULT_X_RSS_SOURCES:
                urls.append(template.format(query=query))

        rss_titles = self._collect_rss_titles(urls, max_items)
        if headlines:
            seen = {re.sub(r"\s+", " ", title).strip() for title in headlines}
            for title in rss_titles:
                normalized = re.sub(r"\s+", " ", title).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                headlines.append(normalized)
                if len(headlines) >= max_items:
                    break
            return headlines[:max_items]

        return rss_titles

    def _headline_keyword_score(self, headlines: list[str]) -> float:
        if not headlines:
            return 0.0

        score = 0.0
        counted = 0
        for headline in headlines:
            text = re.sub(r"\s+", " ", headline.lower()).strip()
            if not text:
                continue
            counted += 1

            positive_hits = sum(1 for token in _POSITIVE_KEYWORDS if token in text)
            negative_hits = sum(1 for token in _NEGATIVE_KEYWORDS if token in text)
            score += float(positive_hits - negative_hits)

        if counted == 0:
            return 0.0

        normalized = math.tanh(score / counted)
        return max(-1.0, min(1.0, normalized))

    def _cached_external_snapshot(self, symbol: str) -> ExternalSentimentSnapshot | None:
        now = datetime.now(timezone.utc)
        key = symbol.upper().strip()
        with self._external_cache_lock:
            entry = self._external_cache.get(key)
        if entry is None:
            return None

        snapshot, expires_at = entry
        if now < expires_at:
            return snapshot
        return None

    def _compute_external_snapshot(self, symbol: str, force_refresh: bool = False) -> ExternalSentimentSnapshot:
        if not self.external_enabled:
            return ExternalSentimentSnapshot(
                score=0.0,
                source="external:disabled",
                fear_greed_index=50.0,
                crypto_news_sentiment=0.0,
                x_sentiment_score=0.0,
                geopolitical_sentiment=0.0,
                event_impact_score=0.0,
            )

        normalized_symbol = symbol.upper().strip()
        if not normalized_symbol:
            return ExternalSentimentSnapshot(
                score=0.0,
                source="external:none",
                fear_greed_index=50.0,
                crypto_news_sentiment=0.0,
                x_sentiment_score=0.0,
                geopolitical_sentiment=0.0,
                event_impact_score=0.0,
            )

        if not force_refresh:
            cached = self._cached_external_snapshot(normalized_symbol)
            if cached is not None:
                return cached

        weighted_total = 0.0
        weight_sum = 0.0
        sources: list[str] = []

        fear_greed_index = self._fetch_fear_greed_index()
        if fear_greed_index is not None:
            fear_greed_score = (fear_greed_index - 50.0) / 50.0
            weighted_total += fear_greed_score * 0.25
            weight_sum += 0.25
            sources.append("fear-greed")
        else:
            fear_greed_index = 50.0

        headlines = self._collect_news_headlines()
        news_score = 0.0
        if headlines:
            news_score = self._headline_keyword_score(headlines)
            weighted_total += news_score * 0.35
            weight_sum += 0.35
            sources.append("news")

            if self.mode == "hf":
                self._ensure_classifier_loaded()

            if self._classifier is not None and self.mode == "hf":
                try:
                    summary = f"{normalized_symbol} market context: " + " | ".join(headlines[:8])
                    hf_result = self._classifier(summary, truncation=True)[0]
                    label = str(hf_result.get("label", "neutral")).lower()
                    confidence = float(hf_result.get("score", 0.5))
                    hf_score = 0.0
                    if "positive" in label:
                        hf_score = confidence
                    elif "negative" in label:
                        hf_score = -confidence

                    weighted_total += max(-1.0, min(1.0, hf_score)) * 0.05
                    weight_sum += 0.05
                    sources.append("hf-news")
                except Exception as exc:
                    logger.warning("HF news sentiment inference failed: %s", exc)

        x_headlines: list[str] = []
        x_score = self._fetch_json_score(
            self.x_sentiment_endpoint or "",
            score_keys=("score", "sentiment", "sentiment_score"),
        )
        if x_score is None:
            x_headlines = self._collect_x_headlines(normalized_symbol)
            if x_headlines:
                x_score = self._headline_keyword_score(x_headlines)
        if x_score is not None:
            weighted_total += x_score * 0.20
            weight_sum += 0.20
            sources.append("x")
        else:
            x_score = 0.0

        geo_headlines: list[str] = []
        geopolitics_score = self._fetch_json_score(
            self.geopolitical_sentiment_endpoint or "",
            score_keys=("score", "sentiment", "sentiment_score"),
        )
        if geopolitics_score is None:
            geo_headlines = self._collect_geopolitical_headlines()
            if geo_headlines:
                geopolitics_score = self._headline_keyword_score(geo_headlines)
        if geopolitics_score is not None:
            weighted_total += geopolitics_score * 0.20
            weight_sum += 0.20
            sources.append("geopolitical")
        else:
            geopolitics_score = 0.0

        event_score = self._event_impact_score(headlines + x_headlines + geo_headlines)

        external_score = weighted_total / weight_sum if weight_sum > 0 else 0.0
        external_score = max(-1.0, min(1.0, external_score))

        source = "external:" + ("+".join(sources) if sources else "none")
        snapshot = ExternalSentimentSnapshot(
            score=external_score,
            source=source,
            fear_greed_index=float(fear_greed_index),
            crypto_news_sentiment=float(news_score),
            x_sentiment_score=float(x_score),
            geopolitical_sentiment=float(geopolitics_score),
            event_impact_score=float(event_score),
        )

        with self._external_cache_lock:
            self._external_cache[normalized_symbol] = (
                snapshot,
                datetime.now(timezone.utc) + timedelta(seconds=self.external_refresh_seconds),
            )

        logger.info(
            "External sentiment snapshot for %s: score=%.4f fng=%.1f news=%.4f x=%.4f geo=%.4f event=%.4f source=%s",
            normalized_symbol,
            snapshot.score,
            snapshot.fear_greed_index,
            snapshot.crypto_news_sentiment,
            snapshot.x_sentiment_score,
            snapshot.geopolitical_sentiment,
            snapshot.event_impact_score,
            snapshot.source,
        )

        return snapshot

    def _compute_external_score(self, symbol: str, force_refresh: bool = False) -> tuple[float, str]:
        snapshot = self._compute_external_snapshot(symbol, force_refresh=force_refresh)
        return snapshot.score, snapshot.source

    def _simple_scores(self, dataframe: pd.DataFrame) -> pd.Series:
        returns = dataframe["close"].pct_change().fillna(0.0)
        short_momentum = returns.rolling(window=5, min_periods=1).mean()
        volatility = returns.rolling(window=20, min_periods=2).std().fillna(1e-4)
        z_score = (short_momentum / (volatility + 1e-6)).clip(lower=-4.0, upper=4.0)
        values = np.tanh(z_score.to_numpy(dtype=np.float64))
        return pd.Series(values, index=dataframe.index, dtype="float64")

    def _hf_score(self, symbol: str, dataframe: pd.DataFrame) -> float:
        self._ensure_classifier_loaded()
        if self._classifier is None:
            return 0.0

        recent = dataframe.tail(30)
        momentum = float(recent["close"].pct_change().fillna(0.0).mean())
        vol = float(recent["close"].pct_change().fillna(0.0).std())

        headline = (
            f"{symbol} short-term momentum is {momentum:.4f} with volatility {vol:.4f}. "
            "Assess crypto market sentiment for this trend."
        )

        result = self._classifier(headline, truncation=True)[0]
        label = str(result.get("label", "neutral")).lower()
        confidence = float(result.get("score", 0.5))

        if "positive" in label:
            return max(-1.0, min(1.0, confidence))
        if "negative" in label:
            return max(-1.0, min(1.0, -confidence))
        return 0.0

    def score_dataframe(
        self,
        symbol: str,
        dataframe: pd.DataFrame,
        force_external_refresh: bool = False,
    ) -> tuple[pd.Series, float, str]:
        baseline = self._simple_scores(dataframe)
        external_scalar, external_source = self._compute_external_score(
            symbol,
            force_refresh=force_external_refresh,
        )

        if self.mode != "hf":
            blended = baseline * 0.75 + external_scalar * 0.25
            return blended.clip(lower=-1.0, upper=1.0), external_scalar, external_source

        try:
            hf_scalar = self._hf_score(symbol, dataframe)
            blended = baseline * 0.55 + hf_scalar * 0.25 + external_scalar * 0.20
            return blended.clip(lower=-1.0, upper=1.0), external_scalar, external_source
        except Exception as exc:
            logger.warning("HF sentiment scoring failed for %s: %s", symbol, exc)
            blended = baseline * 0.75 + external_scalar * 0.25
            return blended.clip(lower=-1.0, upper=1.0), external_scalar, external_source

    def get_external_feature_snapshot(
        self,
        symbol: str,
        force_external_refresh: bool = False,
    ) -> ExternalSentimentSnapshot:
        return self._compute_external_snapshot(symbol, force_refresh=force_external_refresh)

    def score_latest(
        self,
        symbol: str,
        dataframe: pd.DataFrame,
        force_external_refresh: bool = False,
        require_external: bool = False,
    ) -> tuple[float, str, float, str]:
        scores, external_score, external_source = self.score_dataframe(
            symbol,
            dataframe,
            force_external_refresh=force_external_refresh,
        )
        latest = float(scores.iloc[-1]) if not scores.empty else 0.0
        source = f"auto:market+{external_source}"

        has_external_signal = not (
            external_source.endswith("none") or external_source.endswith("disabled")
        )

        if require_external and not has_external_signal:
            raise RuntimeError("No live external sentiment sources are available")

        # Make source explicit when only market signals are available.
        if not has_external_signal:
            source = "auto:market-only"

        # Keep slight influence from external score when dataframe is tiny.
        if len(dataframe) < 10:
            latest = max(-1.0, min(1.0, latest * 0.8 + external_score * 0.2))

        return max(-1.0, min(1.0, latest)), source, external_score, external_source
