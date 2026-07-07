"""CrewAI Multi-Agent Trading Decision System (Streaming).

3 agents: Quant Analyst, Risk Manager, Execution Judge.
Input: market JSON (price, forecast, volatility, sentiment).
Output: TradeDecision Pydantic schema (LONG/SHORT/HOLD + entry/SL/TP/leverage/reasoning).
LLM: DeepSeek via langchain-openai (OpenAI-compatible). Hybrid routing:
  - Quant Analyst & Risk Manager → deepseek-chat (speed/cost)
  - Execution Judge → deepseek-reasoner (deep logic)
Configurable via DEEPSEEK_API_KEY env.

Supports streaming via `run_trading_crew_streaming()` which yields SSE events
in real-time as each agent produces output.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from enum import Enum
from queue import Empty, Queue
from typing import Any, Generator

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TradeAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class TradeDecision(BaseModel):
    """Final trading decision output from the CrewAI pipeline."""

    action: TradeAction = Field(description="LONG, SHORT, or HOLD")
    entry: float = Field(description="Suggested entry price (0 if HOLD)")
    leverage: int = Field(ge=1, le=125, description="Leverage multiplier (1 = spot)")
    stop_loss: float = Field(description="Stop-loss price (0 if HOLD)")
    take_profit: float = Field(description="Take-profit price (0 if HOLD)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score [0,1]")
    reasoning: str = Field(description="Concise human-readable rationale")


class MarketContext(BaseModel):
    """Input context for the CrewAI pipeline."""

    symbol: str = Field(description="Trading pair e.g. BTCUSDT")
    current_price: float
    forecast_median: float = Field(description="Chronos-2 median forecast price")
    forecast_lower: float = Field(description="P5 lower band")
    forecast_upper: float = Field(description="P95 upper band")
    realized_volatility: float = Field(description="Realized vol of last 50 candles")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Blended sentiment [-1,1]")
    funding_rate: float = Field(default=0.0, description="Current funding rate")
    fear_greed_index: float = Field(default=50.0, ge=0.0, le=100.0)
    timeframe: str = Field(default="1h")


class AiAnalyzeRequest(BaseModel):
    """POST /api/ai/analyze request body."""

    symbol: str = Field(description="Trading pair e.g. BTCUSDT")
    timeframe: str = Field(default="1h")


# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------

def _get_deepseek_api_key() -> str:
    """Retrieve DEEPSEEK_API_KEY from environment. Never logs the key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Add it to GitHub Secrets and deploy via SSM."
        )
    return api_key


def _build_chat_llm() -> Any:
    """Build DeepSeek-Chat LLM (fast/cost-optimized).

    Used by Quant Analyst & Risk Manager.
    """
    api_key = _get_deepseek_api_key()
    model = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    logger.info("Initializing DeepSeek Chat LLM: model=%s", model)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.2,
        max_tokens=2048,
    )


def _build_reasoner_llm() -> Any:
    """Build DeepSeek-Reasoner LLM (deep logic/reasoning).

    Used by Execution Judge for final decision making.
    """
    api_key = _get_deepseek_api_key()
    model = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")
    logger.info("Initializing DeepSeek Reasoner LLM: model=%s", model)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
        max_tokens=4096,
    )


def _quant_analyst_agent(llm: Any) -> Any:
    """Reads technical data + Chronos-2 forecast, outputs directional bias."""
    return Agent(
        role="Quant Analyst",
        goal=(
            "Analyze technical indicators, Chronos-2 forecast bands, and realized "
            "volatility to determine a directional bias (LONG/SHORT/NEUTRAL) with "
            "confidence score and optimal entry/exit levels."
        ),
        backstory=(
            "You are a senior quantitative analyst at a top crypto hedge fund. "
            "You specialize in statistical forecasting models and technical analysis. "
            "You are precise, data-driven, and never guess."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _risk_manager_agent(llm: Any) -> Any:
    """Reads sentiment, funding rate, FnG — can veto or reduce exposure."""
    return Agent(
        role="Risk Manager",
        goal=(
            "Evaluate market risk using sentiment, funding rate, and fear/greed index. "
            "Determine if the proposed trade should be blocked, reduced, or approved. "
            "Set appropriate leverage and stop-loss levels."
        ),
        backstory=(
            "You are the Chief Risk Officer of a crypto trading desk. "
            "Your job is to protect capital. You are conservative and skeptical. "
            "If sentiment is extreme or funding rate signals crowded positioning, "
            "you reduce leverage or veto the trade entirely."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _execution_judge_agent(llm: Any) -> Any:
    """Final arbiter — synthesizes Quant + Risk into a single TradeDecision."""
    return Agent(
        role="Execution Judge",
        goal=(
            "Synthesize the Quant Analyst's directional signal and the Risk Manager's "
            "risk assessment into a final trade decision. Output a single JSON object "
            "with action, entry, leverage, stop_loss, take_profit, confidence, reasoning."
        ),
        backstory=(
            "You are the Head of Execution at a systematic trading firm. "
            "You make the final call. You balance opportunity against risk. "
            "Your output must be precise, actionable, and in valid JSON format."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Task Pipeline
# ---------------------------------------------------------------------------

def _build_tasks(
    quant: Any,
    risk: Any,
    judge: Any,
    market: MarketContext,
) -> list[Any]:
    """Create the 3-step task pipeline."""
    market_json = market.model_dump_json(indent=2)

    quant_task = Task(
        description=(
            f"Analyze the following market data and Chronos-2 forecast:\n\n"
            f"```json\n{market_json}\n```\n\n"
            "Provide:\n"
            "1. Directional bias: LONG, SHORT, or NEUTRAL\n"
            "2. Confidence score (0-1)\n"
            "3. Suggested entry price\n"
            "4. Initial stop-loss and take-profit levels\n"
            "5. Brief technical reasoning (2-3 sentences)\n\n"
            "Consider the forecast bands (P5-P95), current price relative to forecast median, "
            "and realized volatility for position sizing."
        ),
        expected_output=(
            "JSON with keys: bias, confidence, entry, stop_loss, take_profit, reasoning"
        ),
        agent=quant,
    )

    risk_task = Task(
        description=(
            "Review the Quant Analyst's trade proposal against risk factors:\n\n"
            f"Market context:\n```json\n{market_json}\n```\n\n"
            "Evaluate:\n"
            "1. Sentiment score: extreme values (> 0.7 or < -0.7) signal caution\n"
            "2. Funding rate: high positive = crowded longs, high negative = crowded shorts\n"
            "3. Fear/Greed Index: extreme fear (< 20) or extreme greed (> 80) = caution\n"
            "4. Realized volatility: high vol = reduce leverage\n\n"
            "Output: approved/reduced/vetoed, adjusted leverage (1-20), adjusted stop-loss, reasoning"
        ),
        expected_output=(
            "JSON with keys: verdict (approved/reduced/vetoed), leverage, stop_loss, reasoning"
        ),
        agent=risk,
        context=[quant_task],
    )

    judge_task = Task(
        description=(
            "You have the Quant Analyst's signal and the Risk Manager's assessment. "
            "Make the final trade decision.\n\n"
            "Output EXACTLY this JSON structure (no markdown, no explanation outside JSON):\n"
            '{\n'
            '  "action": "LONG" | "SHORT" | "HOLD",\n'
            '  "entry": <float>,\n'
            '  "leverage": <int 1-125>,\n'
            '  "stop_loss": <float>,\n'
            '  "take_profit": <float>,\n'
            '  "confidence": <float 0-1>,\n'
            '  "reasoning": "<concise string>"\n'
            "}\n\n"
            "Rules:\n"
            "- If Risk Manager vetoed, action MUST be HOLD with entry/SL/TP = 0\n"
            "- If reduced, apply the Risk Manager's leverage and stop-loss adjustments\n"
            "- Confidence reflects your conviction after both analyses\n"
            "- Reasoning should be 1-2 sentences max"
        ),
        expected_output="A single valid JSON object matching the TradeDecision schema.",
        agent=judge,
        context=[quant_task, risk_task],
    )

    return [quant_task, risk_task, judge_task]


# ---------------------------------------------------------------------------
# Parse final output
# ---------------------------------------------------------------------------

def _parse_trade_decision(raw_output: str) -> TradeDecision:
    """Parse CrewAI's raw text output into a TradeDecision."""
    try:
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(raw_output[json_start:json_end])
        else:
            parsed = json.loads(raw_output)

        return TradeDecision(**parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse CrewAI output: %s | raw: %s", exc, raw_output[:500])
        return TradeDecision(
            action=TradeAction.HOLD,
            entry=0.0,
            leverage=1,
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.0,
            reasoning=f"CrewAI output parsing failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Streaming Public API
# ---------------------------------------------------------------------------

_AGENT_LABELS = {
    "Quant Analyst": "📊 Quant Analyst",
    "Risk Manager": "🛡️ Risk Manager",
    "Execution Judge": "⚖️ Execution Judge",
}

_SENTINEL = object()


def run_trading_crew_streaming(
    market: MarketContext,
) -> Generator[str, None, None]:
    """Execute the 3-agent CrewAI pipeline, yielding SSE events as agents work.

    Yields:
        SSE-formatted strings (`data: ...\n\n`).
        The final event contains `[FINAL_RESULT]:<json>` for the TradeDecision.
    """
    event_queue: Queue[str | object] = Queue()

    def _task_callback(task_output: Any) -> None:
        """Called by CrewAI after each task completes."""
        try:
            agent_name = getattr(task_output, "agent", "")
            if hasattr(agent_name, "role"):
                agent_name = agent_name.role
            label = _AGENT_LABELS.get(str(agent_name), str(agent_name))
            raw = getattr(task_output, "raw", str(task_output))
            event_queue.put(f"[{label}] completed analysis:\n{raw}")
        except Exception as exc:
            logger.debug("task_callback error (non-fatal): %s", exc)
            event_queue.put("[Agent] completed a task step.")

    def _step_callback(step_output: Any) -> None:
        """Called by CrewAI after each agent step (thought/action)."""
        try:
            text = getattr(step_output, "text", None) or str(step_output)
            # Only forward meaningful text, skip very short internal chatter
            if len(text.strip()) > 10:
                event_queue.put(f"💭 {text[:600]}")
        except Exception:
            pass

    def _run_crew() -> None:
        """Background thread: runs the blocking CrewAI pipeline."""
        try:
            # Send progress BEFORE heavy initialization so the client
            # sees activity immediately while the LLM loads.
            event_queue.put(f"🚀 Initializing AI Council for {market.symbol} @ {market.timeframe}...")
            event_queue.put("⏳ Loading DeepSeek LLMs (hybrid routing)...")

            chat_llm = _build_chat_llm()
            reasoner_llm = _build_reasoner_llm()
            event_queue.put("📊 Quant Analyst is analyzing market data...")

            quant = _quant_analyst_agent(chat_llm)
            risk = _risk_manager_agent(chat_llm)
            judge = _execution_judge_agent(reasoner_llm)
            tasks = _build_tasks(quant, risk, judge, market)

            event_queue.put("🛡️ Risk Manager and ⚖️ Execution Judge standing by...")

            crew = Crew(
                agents=[quant, risk, judge],
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
                task_callback=_task_callback,
                step_callback=_step_callback,
            )

            event_queue.put("▶️ Crew kickoff — agents are deliberating...")

            result = crew.kickoff()
            raw_output = str(result)
            decision = _parse_trade_decision(raw_output)

            # CRITICAL: Compact the JSON to a single line with no whitespace.
            # Pydantic's model_dump_json() is usually compact, but the
            # "reasoning" field can contain LLM-generated newlines. We
            # parse and re-dump to guarantee zero newlines in the output,
            # which prevents the SSE line-splitter from fragmenting the
            # JSON across multiple "data:" lines.
            compact_json = json.dumps(
                decision.model_dump(), separators=(",", ":")
            )

            event_queue.put(f"[FINAL_RESULT]:{compact_json}")
        except Exception as exc:
            logger.exception("CrewAI streaming pipeline error")
            event_queue.put(f"[ERROR]:{exc}")
        finally:
            event_queue.put(_SENTINEL)

    # --- CRITICAL: Yield an immediate handshake event BEFORE starting the
    # background thread. This forces Starlette/FastAPI to flush the HTTP
    # response headers (status 200, Content-Type: text/event-stream) to
    # the browser, resolving the "Provisional headers are shown" hang. ---
    yield "data: [CONNECTED]\n\n"

    # Start the blocking crew in a background thread
    thread = threading.Thread(target=_run_crew, daemon=True)
    thread.start()

    # Yield SSE events as they arrive from the queue.
    # Use a short timeout (15s) so keepalives are sent frequently
    # to prevent Caddy/CloudFront/browser from timing out.
    while True:
        try:
            item = event_queue.get(timeout=15)
        except Empty:
            # Keep-alive to prevent proxy/client timeout
            yield "data: [KEEPALIVE]\n\n"
            continue

        if item is _SENTINEL:
            break

        message = str(item)

        # [FINAL_RESULT] and [ERROR] are protocol messages that MUST be
        # yielded as a single "data:" line.  The JSON is already compacted
        # to a single line (no \n), so we bypass the line-splitter.
        if message.startswith("[FINAL_RESULT]:") or message.startswith("[ERROR]:"):
            yield f"data: {message}\n\n"
            continue

        # Regular messages (agent thoughts, progress) — split by newline
        # per SSE spec so multi-line text renders correctly in the terminal.
        for line in message.split("\n"):
            yield f"data: {line}\n"
        yield "\n"


# ---------------------------------------------------------------------------
# Legacy synchronous API (kept for backwards compat / tests)
# ---------------------------------------------------------------------------

def run_trading_crew(market: MarketContext) -> TradeDecision:
    """Execute the 3-agent CrewAI pipeline and return a structured TradeDecision.

    Raises RuntimeError if DEEPSEEK_API_KEY is missing.
    """
    chat_llm = _build_chat_llm()
    reasoner_llm = _build_reasoner_llm()

    quant = _quant_analyst_agent(chat_llm)
    risk = _risk_manager_agent(chat_llm)
    judge = _execution_judge_agent(reasoner_llm)

    tasks = _build_tasks(quant, risk, judge, market)

    crew = Crew(
        agents=[quant, risk, judge],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    logger.info("Running CrewAI trading pipeline for %s @ %s", market.symbol, market.timeframe)
    result = crew.kickoff()

    return _parse_trade_decision(str(result))
