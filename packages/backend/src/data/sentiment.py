from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import re
import threading
import xml.etree.ElementTree as et
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

    async def _fetch_json_score_async(
        self, client: httpx.AsyncClient, url: str, score_keys: tuple[str, ...]
    ) -> float | None:
        if not url:
            return None

        try:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()

            for key in score_keys:
                if key in payload:
                    value = float(payload[key])
                    return max(-1.0, min(1.0, value))
        except Exception as exc:
            logger.warning("External sentiment endpoint failed (%s): %s", url, exc)

        return None

    async def _fetch_fear_greed_index_async(self, client: httpx.AsyncClient) -> float | None:
        url = "https://api.alternative.me/fng/?limit=1"
        try:
            response = await client.get(url)
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

    async def _fetch_news_api_headlines_async(
        self, client: httpx.AsyncClient, symbol: str, max_items: int
    ) -> list[str]:
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
            response = await client.get(self.news_api_endpoint, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()

            return self._extract_headlines_from_payload(payload, max_items)
        except Exception as exc:
            logger.warning("News API fetch failed (%s): %s", self.news_api_endpoint, exc)
            return []

    async def _fetch_x_search_headlines_async(
        self, client: httpx.AsyncClient, query: str, max_items: int
    ) -> list[str]:
        if not self.x_search_endpoint or not self.x_search_bearer_token:
            return []

        params = {"query": query, "max_results": min(100, max(10, int(max_items)))}
        headers = {"Authorization": f"Bearer {self.x_search_bearer_token}"}

        try:
            response = await client.get(self.x_search_endpoint, params=params, headers=headers)
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

    async def _fetch_one_rss_async(self, client: httpx.AsyncClient, url: str) -> list[str]:
        try:
            response = await client.get(url)
            response.raise_for_status()
            xml_text = response.text

            root = et.fromstring(xml_text)
            titles: list[str] = []
            for item in root.findall(".//item/title"):
                title = (item.text or "").strip()
                if title:
                    titles.append(re.sub(r"\s+", " ", title))
            return titles
        except Exception as exc:
            logger.warning("RSS fetch failed (%s): %s", url, exc)
            return []

    async def _collect_rss_titles_async(
        self, client: httpx.AsyncClient, urls: list[str], max_items: int
    ) -> list[str]:
        """Fetch every feed in `urls` concurrently; one slow/dead feed no longer
        blocks the rest (previously a plain sequential `for url in urls` loop)."""
        if not urls:
            return []

        per_feed_results = await asyncio.gather(*(self._fetch_one_rss_async(client, url) for url in urls))

        seen: set[str] = set()
        titles: list[str] = []
        for feed_titles in per_feed_results:
            for title in feed_titles:
                if title in seen:
                    continue
                seen.add(title)
                titles.append(title)
                if len(titles) >= max_items:
                    return titles
        return titles

    async def _collect_news_headlines_async(
        self, client: httpx.AsyncClient, max_headlines: int = 60
    ) -> list[str]:
        # Configured RSS, default crypto RSS, and the news API all just
        # contribute candidate headlines (no fallback semantics between them),
        # so firing them concurrently is a pure speed win with no behavior risk.
        configured_titles, default_titles, api_titles = await asyncio.gather(
            self._collect_rss_titles_async(client, self.news_rss_urls, max_headlines),
            self._collect_rss_titles_async(client, _DEFAULT_CRYPTO_NEWS_RSS_URLS, max_headlines),
            self._fetch_news_api_headlines_async(client, "crypto", max_headlines),
        )

        seen: set[str] = set()
        headlines: list[str] = []
        for group in (configured_titles, default_titles, api_titles):
            for title in group:
                normalized = re.sub(r"\s+", " ", title).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                headlines.append(normalized)
                if len(headlines) >= max_headlines:
                    return headlines
        return headlines

    async def _resolve_geo_signal_async(
        self, client: httpx.AsyncClient, max_headlines: int = 20
    ) -> tuple[float | None, list[str]]:
        direct_score = await self._fetch_json_score_async(
            client,
            self.geopolitical_sentiment_endpoint or "",
            score_keys=("score", "sentiment", "sentiment_score"),
        )
        if direct_score is not None:
            return direct_score, []
        if self.geopolitical_sentiment_endpoint:
            # Endpoint was configured but failed — do not fall back to RSS
            # scraping, matching the original sync behavior.
            return None, []

        headlines = await self._collect_rss_titles_async(client, _DEFAULT_GEOPOLITICAL_RSS_URLS, max_headlines)
        if headlines:
            return self._headline_keyword_score(headlines), headlines
        return None, []

    async def _resolve_x_signal_async(
        self, client: httpx.AsyncClient, symbol: str, max_items: int = 40
    ) -> tuple[float | None, list[str]]:
        direct_score = await self._fetch_json_score_async(
            client,
            self.x_sentiment_endpoint or "",
            score_keys=("score", "sentiment", "sentiment_score"),
        )
        if direct_score is not None:
            return direct_score, []
        if self.x_sentiment_endpoint:
            # Endpoint was configured but failed — do not fall back to nitter
            # scraping, matching the original sync behavior.
            return None, []

        symbol_token = symbol.upper().replace("USDT", "").replace("USD", "")
        topics = [symbol_token or symbol.upper(), *_DEFAULT_X_TOPICS]
        if self.event_keywords:
            topics.extend(self.event_keywords[:6])

        # _DEFAULT_X_TOPICS is a static non-empty list, so x_query is always
        # truthy in practice; _fetch_x_search_headlines_async already no-ops
        # safely if the search endpoint/token aren't configured.
        x_query = self.x_search_query or " OR ".join(sorted(set(topics)))
        search_query = f"{x_query} lang:en" if x_query else ""

        urls: list[str] = []
        for topic in topics:
            query = quote_plus(f"{topic} lang:en")
            for template in _DEFAULT_X_RSS_SOURCES:
                urls.append(template.format(query=query))

        search_headlines, rss_titles = await asyncio.gather(
            self._fetch_x_search_headlines_async(client, search_query, max_items),
            self._collect_rss_titles_async(client, urls, max_items),
        )

        headlines = list(search_headlines)
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
        else:
            headlines = rss_titles

        if headlines:
            return self._headline_keyword_score(headlines), headlines
        return None, []

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
        now = datetime.now(UTC)
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

        # All independent external fetches (Fear&Greed, news RSS/API, X,
        # geopolitical) run concurrently via asyncio.gather inside
        # _gather_external_signals_async — previously each ran sequentially,
        # so the total latency was the SUM of every source's round-trip
        # instead of the MAX of the slowest one.
        #
        # This method is reached from BOTH a sync context (/predict, run in a
        # plain threadpool worker with no event loop) AND an async context
        # (/api/ai/analyze's `async def` route, which calls this inline while
        # its own event loop is already running). A bare asyncio.run() here
        # crashes with "cannot be called from a running event loop" in the
        # second case. Submitting to a dedicated one-off thread guarantees a
        # loop-free thread every time, regardless of the caller's context. A
        # fresh executor per call (not a shared persistent one) is
        # deliberate: it lets concurrent requests each get their own isolated
        # thread instead of serializing behind a single shared worker.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._gather_external_signals_async(normalized_symbol))
            (
                fear_greed_index,
                headlines,
                (x_score, x_headlines),
                (geopolitics_score, geo_headlines),
            ) = future.result()

        weighted_total = 0.0
        weight_sum = 0.0
        sources: list[str] = []

        if fear_greed_index is not None:
            fear_greed_score = (fear_greed_index - 50.0) / 50.0
            weighted_total += fear_greed_score * 0.25
            weight_sum += 0.25
            sources.append("fear-greed")
        else:
            fear_greed_index = 50.0

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

        # x_score/x_headlines and geopolitics_score/geo_headlines were already
        # resolved (direct endpoint preferred, RSS fallback only if
        # unconfigured) inside _gather_external_signals_async above.
        if x_score is not None:
            weighted_total += x_score * 0.20
            weight_sum += 0.20
            sources.append("x")
        else:
            x_score = 0.0

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
                datetime.now(UTC) + timedelta(seconds=self.external_refresh_seconds),
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

    async def _gather_external_signals_async(
        self, normalized_symbol: str
    ) -> tuple[float | None, list[str], tuple[float | None, list[str]], tuple[float | None, list[str]]]:
        """Fire Fear&Greed, news, X, and geopolitical sentiment fetches concurrently.

        Returns (fear_greed_index, headlines, (x_score, x_headlines),
        (geopolitics_score, geo_headlines)) — the same raw ingredients the
        original sequential implementation computed one at a time.
        """
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=True) as client:
            return await asyncio.gather(
                self._fetch_fear_greed_index_async(client),
                self._collect_news_headlines_async(client),
                self._resolve_x_signal_async(client, normalized_symbol),
                self._resolve_geo_signal_async(client),
            )

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
            external_source.endswith(("none", "disabled"))
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
