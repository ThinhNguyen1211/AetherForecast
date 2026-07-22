"""CrewAI Multi-Agent Trading Decision System (Streaming).

4 agents: Quant Analyst, Devil's Advocate, Risk Manager, Execution Judge.
Input: market JSON (price, forecast, volatility, sentiment).
Output: AiCouncilDecision Pydantic schema (LONG/SHORT/HOLD + entry/SL/TP/leverage/reasoning).
LLM: DeepSeek via langchain-openai (OpenAI-compatible). Hybrid routing:
  - Quant Analyst, Devil's Advocate & Risk Manager → deepseek-chat (speed/cost)
  - Execution Judge → deepseek-reasoner (deep logic/reasoning)
Configurable via DEEPSEEK_API_KEY env.

Supports streaming via `run_trading_crew_streaming()` which yields SSE events
in real-time as each agent produces output.
"""

import json
import logging
import os
import threading
import traceback
from enum import Enum
from queue import Empty, Queue
from typing import Any, Generator, Literal

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


class RiskProfile(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    DEGEN = "DEGEN"


class AiCouncilDecision(BaseModel):
    """Final pro-trader decision output from the CrewAI pipeline."""

    action: TradeAction = Field(description="LONG, SHORT, or HOLD")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score [0,1]")
    entry: float = Field(default=0.0, description="Suggested entry price (0 if HOLD)")
    entry_condition: str = Field(
        default="Market Order",
        description="How to enter, e.g. 'Market Order' or 'Wait for 1H close above X'",
    )
    leverage: int = Field(ge=0, le=125, default=0, description="Leverage multiplier (0 if HOLD, 1 = spot)")
    position_size_pct: float = Field(
        ge=0.0, le=100.0, default=0.0, description="Position size as percent of portfolio (e.g. 2.0 = 2%)"
    )
    stop_loss: float = Field(default=0.0, description="Stop-loss price (0 if HOLD)")
    invalidation_point: str = Field(
        default="",
        description="Macro/technical condition that invalidates the trade, e.g. 'Close if DXY breaks 106'",
    )
    take_profit_1: float = Field(default=0.0, description="Safe take-profit for 50% of position (0 if HOLD)")
    take_profit_2: float = Field(default=0.0, description="Aggressive take-profit for remaining 50% (0 if HOLD)")
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
    risk_profile: RiskProfile = Field(
        default=RiskProfile.BALANCED, description="Trader risk profile"
    )
    language: Literal["en", "vi"] = Field(
        default="vi", description="Language for the final reasoning field"
    )


# Force Pydantic V2 to fully resolve these models before they are used by
# FastAPI route TypeAdapters or CrewAI task outputs. Prevents ForwardRef crashes.
AiCouncilDecision.model_rebuild()
MarketContext.model_rebuild()


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


def _normalize_deepseek_model(model: str) -> str:
    """Ensure DeepSeek model names carry the LiteLLM provider prefix.

    CrewAI uses LiteLLM under the hood, which requires 'provider/model'.
    Without the prefix LiteLLM raises:
      BadRequestError: LLM Provider NOT provided. You passed model=deepseek-chat
    """
    model = model.strip()
    if model and "/" not in model and model.startswith("deepseek-"):
        return f"deepseek/{model}"
    return model


def _build_chat_llm() -> Any:
    """Build DeepSeek-Chat LLM (fast/cost-optimized).

    Used by Quant Analyst, Devil's Advocate, and Risk Manager.
    """
    api_key = _get_deepseek_api_key()
    model = _normalize_deepseek_model(
        os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek/deepseek-chat")
    )
    logger.info("Initializing DeepSeek Chat LLM: model=%s", model)

    return ChatOpenAI(
        model_name=model,
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.2,
        max_tokens=2048,
    )


def _build_reasoner_llm() -> Any:
    """Build DeepSeek-Reasoner LLM (deep logic/reasoning).

    Used by Execution Judge for final decision making.
    """
    api_key = _get_deepseek_api_key()
    model = _normalize_deepseek_model(
        os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek/deepseek-reasoner")
    )
    logger.info("Initializing DeepSeek Reasoner LLM: model=%s", model)

    return ChatOpenAI(
        model_name=model,
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1",
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


def _devils_advocate_agent(llm: Any) -> Any:
    """Contrarian analyst that attacks the Quant Analyst's signal to fight confirmation bias."""
    return Agent(
        role="Devil's Advocate",
        goal=(
            "Ruthlessly attack the Quant Analyst's directional signal. Find the strongest "
            "technical, macro, and contextual reasons why the proposed trade will fail. "
            "Expose confirmation bias, flawed assumptions, and hidden risks."
        ),
        backstory=(
            "You are a legendary short-seller and risk analyst. Your sole purpose is to "
            "challenge bullish or bearish narratives. You are cynical, evidence-driven, and "
            "never accept face-value arguments. You pride yourself on spotting the one reason "
            "a trade will blow up."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _risk_manager_agent(llm: Any, risk_profile: RiskProfile) -> Any:
    """Reads sentiment, funding rate, FnG — can veto or reduce exposure."""
    leverage_rule = {
        RiskProfile.CONSERVATIVE: "Leverage MUST be between 2x and 10x (inclusive).",
        RiskProfile.BALANCED: "Leverage MUST be between 11x and 40x (inclusive).",
        RiskProfile.DEGEN: "Leverage MUST be between 41x and 125x (inclusive).",
    }[risk_profile]

    return Agent(
        role="Risk Manager",
        goal=(
            "Evaluate market risk using sentiment, funding rate, and fear/greed index. "
            "Determine if the proposed trade should be blocked, reduced, or approved. "
            "Set appropriate leverage, stop-loss, position size, and invalidation point."
        ),
        backstory=(
            "You are the Chief Risk Officer of a crypto trading desk. "
            "Your job is to protect capital. You are conservative and skeptical. "
            "If sentiment is extreme or funding rate signals crowded positioning, "
            "you reduce leverage or veto the trade entirely. "
            f"The current trader risk profile is {risk_profile.value}. STRICT RULE: {leverage_rule} "
            "Never recommend leverage outside this range. Also define position_size_pct, "
            "invalidation_point, and split take-profit levels (take_profit_1 safe, take_profit_2 aggressive)."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _execution_judge_agent(llm: Any) -> Any:
    """Final arbiter — synthesizes Quant + Risk into a single AiCouncilDecision."""
    return Agent(
        role="Execution Judge",
        goal=(
            "Synthesize the Quant Analyst's directional signal and the Risk Manager's "
            "risk assessment into a final pro-trader decision. Output a single JSON object "
            "with action, confidence, entry, entry_condition, leverage, position_size_pct, "
            "stop_loss, invalidation_point, take_profit_1, take_profit_2, reasoning."
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
    devil: Any,
    risk: Any,
    judge: Any,
    market: MarketContext,
) -> list[Any]:
    """Create the 4-step task pipeline: Quant -> Devil -> Risk -> Judge."""
    market_json = market.model_dump_json(indent=2)
    risk_profile = market.risk_profile

    quant_task = Task(
        description=(
            f"Analyze the following market data and Chronos-2 forecast (Risk Profile: {risk_profile.value}):\n\n"
            f"```json\n{market_json}\n```\n\n"
            "Provide:\n"
            "1. Directional bias: LONG, SHORT, or NEUTRAL\n"
            "2. Confidence score (0-1)\n"
            "3. Suggested entry price\n"
            "4. Entry condition: how to enter, e.g. 'Market Order' or 'Wait for 1H close above X'\n"
            "5. Initial stop-loss price\n"
            "6. Two take-profit levels: take_profit_1 (safe/conservative) and take_profit_2 (aggressive/moon)\n"
            "7. Invalidation point: macro/technical condition that would invalidate the setup\n"
            "8. Brief technical reasoning (2-3 sentences)\n\n"
            "Consider the forecast bands (P5-P95), current price relative to forecast median, "
            "and realized volatility for position sizing."
        ),
        expected_output=(
            "JSON with keys: bias, confidence, entry, entry_condition, stop_loss, "
            "take_profit_1, take_profit_2, invalidation_point, reasoning"
        ),
        agent=quant,
    )

    devils_advocate_task = Task(
        description=(
            "You are the Devil's Advocate. Review the Quant Analyst's signal and "
            "ruthlessly attack it. Your job is to find the strongest reasons the trade will fail.\n\n"
            "Provide:\n"
            "1. The strongest 2-3 technical or macro reasons the Quant's signal is wrong\n"
            "2. Any hidden risks not obvious in the forecast bands or sentiment\n"
            "3. Conditions under which the trade idea would be invalidated\n"
            "4. A final verdict: weak signal / strong contradiction / needs more confirmation\n"
            "5. Brief contrarian reasoning (2-3 sentences)\n\n"
            "Be skeptical and evidence-driven, not contrarian for its own sake."
        ),
        expected_output=(
            "JSON with keys: contradictions (list), hidden_risks (list), invalidation_conditions, "
            "verdict (weak_signal/strong_contradiction/needs_confirmation), reasoning"
        ),
        agent=devil,
        context=[quant_task],
    )

    risk_task = Task(
        description=(
            "Review the Quant Analyst's trade proposal AND the Devil's Advocate critique "
            f"against risk factors. The trader's risk profile is {risk_profile.value}.\n\n"
            f"Market context:\n```json\n{market_json}\n```\n\n"
            "Evaluate:\n"
            "1. Sentiment score: extreme values (> 0.7 or < -0.7) signal caution\n"
            "2. Funding rate: high positive = crowded longs, high negative = crowded shorts\n"
            "3. Fear/Greed Index: extreme fear (< 20) or extreme greed (> 80) = caution\n"
            "4. Realized volatility: high vol = reduce leverage / position size\n"
            "5. Devil's Advocate contradictions: if strong, veto or reduce aggressively\n\n"
            "STRICT LEVERAGE BOUNDS:\n"
            "- CONSERVATIVE: leverage must be 2x-10x\n"
            "- BALANCED: leverage must be 11x-40x\n"
            "- DEGEN: leverage must be 41x-125x\n\n"
            "Output: approved/reduced/vetoed, adjusted leverage (respect bounds), "
            "adjusted stop-loss, position_size_pct, invalidation_point, take_profit_1, take_profit_2, reasoning"
        ),
        expected_output=(
            "JSON with keys: verdict (approved/reduced/vetoed), leverage, stop_loss, "
            "position_size_pct, invalidation_point, take_profit_1, take_profit_2, reasoning"
        ),
        agent=risk,
        context=[quant_task, devils_advocate_task],
    )

    judge_task = Task(
        description=(
            "You have the Quant Analyst's signal, the Devil's Advocate critique, "
            "and the Risk Manager's assessment. "
            f"The trader's risk profile is {risk_profile.value}. "
            "Make the final pro-trader trade decision.\n\n"
            "Output EXACTLY this JSON structure (no markdown, no explanation outside JSON):\n"
            '{\n'
            '  "action": "LONG" | "SHORT" | "HOLD",\n'
            '  "confidence": <float 0-1>,\n'
            '  "entry": <float>,\n'
            '  "entry_condition": "<e.g. Market Order or Wait for 1H close above X>",\n'
            '  "leverage": <int within profile bounds>,\n'
            '  "position_size_pct": <float e.g. 2.0 for 2%>,\n'
            '  "stop_loss": <float>,\n'
            '  "invalidation_point": "<macro/technical invalidation>",\n'
            '  "take_profit_1": <float>,\n'
            '  "take_profit_2": <float>,\n'
            '  "reasoning": "<concise string>"\n'
            "}\n\n"
            "Rules:\n"
            "- If Risk Manager vetoed, action MUST be HOLD with entry/SL/TPs/leverage = 0\n"
            "- If the Devil's Advocate raised strong contradictions and Risk Manager did not veto, "
            "  consider HOLD or reduce leverage/confidence accordingly\n"
            "- If reduced, apply the Risk Manager's leverage and stop-loss adjustments\n"
            "- Leverage MUST respect the risk profile bounds\n"
            "- CRITICAL: If action is HOLD, all price targets (entry, stop_loss, take_profit_1, take_profit_2) "
            "  MUST be 0.0, BUT the 'confidence' score must reflect your certainty in the HOLD decision "
            "  (e.g., 0.85). Do NOT output confidence = 0.0 for a HOLD decision.\n"
            "- Confidence reflects your conviction after weighing Quant + Devil's Advocate + Risk Manager\n"
            "- Reasoning should be 1-2 sentences max\n"
            f"- The final 'reasoning' field in the JSON MUST be written in {market.language} "
            "(if 'vi' use Vietnamese, if 'en' use English). All other keys and processes remain in English."
        ),
        expected_output="A single valid JSON object matching the AiCouncilDecision schema.",
        agent=judge,
        context=[quant_task, devils_advocate_task, risk_task],
    )

    return [quant_task, devils_advocate_task, risk_task, judge_task]


# ---------------------------------------------------------------------------
# Parse final output
# ---------------------------------------------------------------------------

def _parse_trade_decision(raw_output: str) -> AiCouncilDecision:
    """Parse CrewAI's raw text output into an AiCouncilDecision."""
    try:
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(raw_output[json_start:json_end])
        else:
            parsed = json.loads(raw_output)

        return AiCouncilDecision(**parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse CrewAI output: %s | raw: %s", exc, raw_output[:500])
        return AiCouncilDecision(
            action=TradeAction.HOLD,
            confidence=0.0,
            entry=0.0,
            entry_condition="",
            leverage=0,
            position_size_pct=0.0,
            stop_loss=0.0,
            invalidation_point="",
            take_profit_1=0.0,
            take_profit_2=0.0,
            reasoning=f"CrewAI output parsing failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Streaming Public API
# ---------------------------------------------------------------------------

_AGENT_LABELS = {
    "Quant Analyst": "📊 Quant Analyst",
    "Devil's Advocate": "😈 Devil's Advocate",
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
        The final event contains `[FINAL_RESULT]:<json>` for the AiCouncilDecision.
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
        """Background thread: runs the blocking CrewAI pipeline.

        Every possible failure point is wrapped so that the error is
        always surfaced to the frontend via the event queue — never
        swallowed silently.
        """
        try:
            # --- Phase 1: LLM Initialization ---
            event_queue.put(f"🚀 Initializing AI Council for {market.symbol} @ {market.timeframe}...")
            event_queue.put("⏳ Loading DeepSeek LLMs (hybrid routing)...")

            try:
                chat_llm = _build_chat_llm()
                reasoner_llm = _build_reasoner_llm()
            except Exception as llm_exc:
                logger.exception("DeepSeek LLM initialization failed")
                error_tb = traceback.format_exc().replace("\n", " | ")
                event_queue.put(
                    f"[ERROR]:DeepSeek LLM Init Failed - {type(llm_exc).__name__}: {llm_exc}"
                )
                event_queue.put(f"[TRACE]:{error_tb}")
                return

            event_queue.put("✅ LLMs loaded. Building agent team...")

            # --- Phase 2: Agent & Task Construction ---
            quant = _quant_analyst_agent(chat_llm)
            devil = _devils_advocate_agent(chat_llm)
            risk = _risk_manager_agent(chat_llm, market.risk_profile)
            judge = _execution_judge_agent(reasoner_llm)
            tasks = _build_tasks(quant, devil, risk, judge, market)

            event_queue.put("📊 Quant Analyst is analyzing market data...")
            event_queue.put("😈 Devil's Advocate is stress-testing the signal...")
            event_queue.put("🛡️ Risk Manager and ⚖️ Execution Judge standing by...")

            crew = Crew(
                agents=[quant, devil, risk, judge],
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
                task_callback=_task_callback,
                step_callback=_step_callback,
            )

            # --- Phase 3: Crew Execution ---
            event_queue.put("▶️ Crew kickoff — agents are deliberating...")

            try:
                result = crew.kickoff()
            except Exception as crew_exc:
                logger.exception("CrewAI kickoff crashed")
                error_tb = traceback.format_exc().replace("\n", " | ")
                event_queue.put(
                    f"[ERROR]:CrewAI Crash - {type(crew_exc).__name__}: {crew_exc}"
                )
                event_queue.put(f"[TRACE]:{error_tb}")
                return

            # --- Phase 4: Parse & Yield Result ---
            raw_output = str(result)
            decision = _parse_trade_decision(raw_output)

            # Compact the JSON to a single line — no newlines allowed.
            compact_json = json.dumps(
                decision.model_dump(), separators=(",", ":")
            )

            event_queue.put(f"[FINAL_RESULT]:{compact_json}")

        except Exception as exc:
            # Catch-all for anything we didn't anticipate above.
            logger.exception("CrewAI streaming pipeline error (unexpected)")
            error_tb = traceback.format_exc().replace("\n", " | ")
            event_queue.put(
                f"[ERROR]:CrewAI Unexpected Error - {type(exc).__name__}: {exc}"
            )
            event_queue.put(f"[TRACE]:{error_tb}")
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
    # The entire yield loop is wrapped in try/except so that any crash
    # in the generator itself (queue errors, serialization bugs) is
    # surfaced to the client instead of silently dropping the stream.
    try:
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
            if message.startswith("[FINAL_RESULT]:") or message.startswith("[ERROR]:") or message.startswith("[TRACE]:"):
                yield f"data: {message}\n\n"
                continue

            # Regular messages (agent thoughts, progress) — split by newline
            # per SSE spec so multi-line text renders correctly in the terminal.
            for line in message.split("\n"):
                yield f"data: {line}\n"
            yield "\n"

    except Exception as gen_exc:
        # If the generator itself crashes, surface it to the client.
        logger.exception("SSE generator crash")
        error_tb = traceback.format_exc().replace("\n", " | ")
        yield f"data: [ERROR]:SSE Stream Error - {type(gen_exc).__name__}: {gen_exc}\n\n"
        yield f"data: [TRACE]:{error_tb}\n\n"


# ---------------------------------------------------------------------------
# Legacy synchronous API (kept for backwards compat / tests)
# ---------------------------------------------------------------------------

def run_trading_crew(market: MarketContext) -> AiCouncilDecision:
    """Execute the 3-agent CrewAI pipeline and return a structured AiCouncilDecision.

    Raises RuntimeError if DEEPSEEK_API_KEY is missing.
    """
    chat_llm = _build_chat_llm()
    reasoner_llm = _build_reasoner_llm()

    quant = _quant_analyst_agent(chat_llm)
    devil = _devils_advocate_agent(chat_llm)
    risk = _risk_manager_agent(chat_llm, market.risk_profile)
    judge = _execution_judge_agent(reasoner_llm)

    tasks = _build_tasks(quant, devil, risk, judge, market)

    crew = Crew(
        agents=[quant, devil, risk, judge],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    logger.info("Running CrewAI trading pipeline for %s @ %s", market.symbol, market.timeframe)
    result = crew.kickoff()

    return _parse_trade_decision(str(result))
