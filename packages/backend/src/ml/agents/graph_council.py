"""LangGraph-based cyclic AI Council debate engine.

This module replaces CrewAI's Process.sequential with a stateful graph where
the Devil's Advocate can force the Quant Analyst to re-evaluate its signal if
severe contradictions are detected. The graph prevents infinite loops via an
explicit iteration_count cap.
"""

import json
import logging
import re
import threading
import traceback
from collections.abc import Generator
from queue import Empty, Queue
from typing import Any, TypedDict

from crewai import Task
from langgraph.graph import END, StateGraph

from src.ml.agents.crew import (
    AiCouncilDecision,
    MarketContext,
    RiskProfile,
    _build_chat_llm,
    _build_reasoner_llm,
    _devils_advocate_agent,
    _execution_judge_agent,
    _quant_analyst_agent,
    _risk_manager_agent,
)

logger = logging.getLogger(__name__)

MAX_DEBATE_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class AiCouncilState(TypedDict):
    """Shared state carried through the LangGraph debate graph."""

    market_data: dict[str, Any]
    quant_signal: dict[str, Any]
    devil_critique: dict[str, Any]
    risk_assessment: dict[str, Any]
    final_decision: dict[str, Any]
    iteration_count: int
    event_queue: Queue[str | object] | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_json_from_markdown(raw: str) -> str:
    """Strip a ```json ... ``` / ``` ... ``` markdown code fence from LLM output.

    DeepSeek (and other chat models) frequently wrap JSON responses in a
    markdown code fence despite prompt instructions not to. json.loads() fails
    immediately on the leading backticks (`Expecting value: line 1 column 1`).
    If no fence is present, the input is returned unchanged (stripped).
    """
    match = _MARKDOWN_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


def _parse_agent_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from agent output, tolerating markdown code fences."""
    cleaned = extract_json_from_markdown(raw)
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(cleaned[json_start:json_end])
    return json.loads(cleaned)


def _market_context_from_state(state: AiCouncilState) -> MarketContext:
    """Rehydrate a MarketContext from the graph state."""
    data = state["market_data"]
    return MarketContext(**data)


def _emit_info(state: AiCouncilState, message: str) -> None:
    """Push a streaming info event to the frontend terminal if a queue exists."""
    queue = state.get("event_queue")
    if queue is not None:
        queue.put(f"[INFO] {message}")


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def quant_analyst_node(state: AiCouncilState) -> dict[str, Any]:
    """Quant Analyst evaluates the market and proposes a directional signal."""
    market = _market_context_from_state(state)
    market_json = json.dumps(state["market_data"], indent=2)
    iteration = state.get("iteration_count", 0) + 1

    _emit_info(
        state,
        f"📊 Quant Analyst is evaluating the market (Iteration {iteration}/{MAX_DEBATE_ITERATIONS})",
    )

    prompt = (
        f"Analyze the following market data and Chronos-2 forecast "
        f"(Risk Profile: {market.risk_profile.value}, Iteration: {iteration}):\n\n"
        f"```json\n{market_json}\n```\n\n"
    )

    if iteration > 1 and state.get("devil_critique"):
        _emit_info(
            state,
            "🔄 Devil's Advocate forced a re-evaluation; Quant Analyst is addressing contradictions",
        )
        critique_json = json.dumps(state["devil_critique"], indent=2)
        prompt += (
            "\nThe Devil's Advocate raised the following critique in the previous round. "
            "You MUST address these contradictions and either revise your signal or "
            "strengthen your reasoning:\n\n"
            f"```json\n{critique_json}\n```\n\n"
        )

    prompt += (
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
    )

    agent = _quant_analyst_agent(_build_chat_llm())
    task = Task(
        description=prompt,
        expected_output=(
            "JSON with keys: bias, confidence, entry, entry_condition, stop_loss, "
            "take_profit_1, take_profit_2, invalidation_point, reasoning"
        ),
        agent=agent,
    )
    raw = agent.execute_task(task)

    try:
        quant_signal = _parse_agent_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Quant Analyst output parse failed: %s | raw: %s", exc, raw[:500])
        quant_signal = {
            "bias": "NEUTRAL",
            "confidence": 0.0,
            "entry": 0.0,
            "entry_condition": "",
            "stop_loss": 0.0,
            "take_profit_1": 0.0,
            "take_profit_2": 0.0,
            "invalidation_point": "",
            "reasoning": f"Quant parse failed: {exc}",
        }

    return {
        "quant_signal": quant_signal,
        "iteration_count": iteration,
    }


def devils_advocate_node(state: AiCouncilState) -> dict[str, Any]:
    """Devil's Advocate attacks the Quant Analyst's signal."""
    _emit_info(state, "😈 Devil's Advocate is stress-testing the Quant signal")
    market_json = json.dumps(state["market_data"], indent=2)
    quant_json = json.dumps(state["quant_signal"], indent=2)

    prompt = (
        "You are the Devil's Advocate. Review the Quant Analyst's signal and "
        "ruthlessly attack it. Your job is to find the strongest reasons the trade will fail.\n\n"
        f"Market context:\n```json\n{market_json}\n```\n\n"
        f"Quant Analyst signal:\n```json\n{quant_json}\n```\n\n"
        "Provide:\n"
        "1. The strongest 2-3 technical or macro reasons the Quant's signal is wrong\n"
        "2. Any hidden risks not obvious in the forecast bands or sentiment\n"
        "3. Conditions under which the trade idea would be invalidated\n"
        "4. A final verdict: weak_signal / strong_contradiction / needs_confirmation\n"
        "5. Brief contrarian reasoning (2-3 sentences)\n\n"
        "Be skeptical and evidence-driven, not contrarian for its own sake.\n\n"
        "CRITICAL OUTPUT FORMAT: Respond with ONE raw JSON object and NOTHING else. "
        "Do NOT wrap it in a markdown code fence (no ```json or ```). Do NOT include any "
        "explanation, heading, or text before or after the JSON. Output EXACTLY this shape:\n"
        "{\n"
        '  "contradictions": ["<reason 1>", "<reason 2>"],\n'
        '  "hidden_risks": ["<risk 1>"],\n'
        '  "invalidation_conditions": "<condition>",\n'
        '  "verdict": "weak_signal" | "strong_contradiction" | "needs_confirmation",\n'
        '  "reasoning": "<2-3 sentences>"\n'
        "}"
    )

    agent = _devils_advocate_agent(_build_chat_llm())
    task = Task(
        description=prompt,
        expected_output=(
            "A single raw JSON object (no markdown fence) with keys: contradictions (list), "
            "hidden_risks (list), invalidation_conditions, "
            "verdict (weak_signal/strong_contradiction/needs_confirmation), reasoning"
        ),
        agent=agent,
    )
    raw = agent.execute_task(task)

    try:
        devil_critique = _parse_agent_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        # An unparsable critique is NOT evidence that the Quant signal is sound —
        # silently defaulting to "needs_confirmation" would swallow the failure and
        # let a potentially-bad trade sail through to the Risk Manager unchallenged.
        # Force a retry by routing back to the Quant Analyst instead (route_after_devil
        # treats "strong_contradiction" as the trigger, capped by MAX_DEBATE_ITERATIONS).
        logger.critical(
            "Devil's Advocate JSON parse failed even after markdown-fence stripping: %s | raw: %s",
            exc,
            raw[:500],
        )
        _emit_info(
            state,
            "⚠️ Devil's Advocate output could not be parsed — forcing a re-evaluation as a precaution",
        )
        devil_critique = {
            "contradictions": [],
            "hidden_risks": [],
            "invalidation_conditions": "",
            "verdict": "strong_contradiction",
            "reasoning": f"Devil's Advocate output could not be parsed as JSON: {exc}",
        }

    return {"devil_critique": devil_critique}


def risk_manager_node(state: AiCouncilState) -> dict[str, Any]:
    """Risk Manager evaluates the signal against risk factors and critique."""
    _emit_info(state, "🛡️ Risk Manager is weighing signal against risk factors")
    market = _market_context_from_state(state)
    market_json = json.dumps(state["market_data"], indent=2)
    quant_json = json.dumps(state["quant_signal"], indent=2)
    critique_json = json.dumps(state["devil_critique"], indent=2)

    leverage_rule = {
        RiskProfile.CONSERVATIVE: "2x-10x",
        RiskProfile.BALANCED: "11x-40x",
        RiskProfile.DEGEN: "41x-125x",
    }[market.risk_profile]

    prompt = (
        "Review the Quant Analyst's trade proposal AND the Devil's Advocate critique "
        f"against risk factors. The trader's risk profile is {market.risk_profile.value}.\n\n"
        f"Market context:\n```json\n{market_json}\n```\n\n"
        f"Quant Analyst signal:\n```json\n{quant_json}\n```\n\n"
        f"Devil's Advocate critique:\n```json\n{critique_json}\n```\n\n"
        "Evaluate:\n"
        "1. Sentiment score: extreme values (> 0.7 or < -0.7) signal caution\n"
        "2. Funding rate: high positive = crowded longs, high negative = crowded shorts\n"
        "3. Fear/Greed Index: extreme fear (< 20) or extreme greed (> 80) = caution\n"
        "4. Realized volatility: high vol = reduce leverage / position size\n"
        "5. Devil's Advocate contradictions: if strong, veto or reduce aggressively\n\n"
        f"STRICT LEVERAGE BOUNDS: {leverage_rule}\n\n"
        "Output: approved/reduced/vetoed, adjusted leverage (respect bounds), "
        "adjusted stop-loss, position_size_pct, invalidation_point, take_profit_1, take_profit_2, reasoning"
    )

    agent = _risk_manager_agent(_build_chat_llm(), market.risk_profile)
    task = Task(
        description=prompt,
        expected_output=(
            "JSON with keys: verdict (approved/reduced/vetoed), leverage, stop_loss, "
            "position_size_pct, invalidation_point, take_profit_1, take_profit_2, reasoning"
        ),
        agent=agent,
    )
    raw = agent.execute_task(task)

    try:
        risk_assessment = _parse_agent_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Risk Manager output parse failed: %s | raw: %s", exc, raw[:500])
        risk_assessment = {
            "verdict": "vetoed",
            "leverage": 0,
            "stop_loss": 0.0,
            "position_size_pct": 0.0,
            "invalidation_point": "",
            "take_profit_1": 0.0,
            "take_profit_2": 0.0,
            "reasoning": f"Risk Manager parse failed: {exc}",
        }

    return {"risk_assessment": risk_assessment}


def execution_judge_node(state: AiCouncilState) -> dict[str, Any]:
    """Execution Judge synthesizes all inputs into the final AiCouncilDecision."""
    _emit_info(state, "⚖️ Execution Judge is rendering the final decision")
    market = _market_context_from_state(state)
    market_json = json.dumps(state["market_data"], indent=2)
    quant_json = json.dumps(state["quant_signal"], indent=2)
    critique_json = json.dumps(state["devil_critique"], indent=2)
    risk_json = json.dumps(state["risk_assessment"], indent=2)

    prompt = (
        "You have the Quant Analyst's signal, the Devil's Advocate critique, "
        f"and the Risk Manager's assessment. The trader's risk profile is {market.risk_profile.value}. "
        "Make the final pro-trader trade decision.\n\n"
        f"Market context:\n```json\n{market_json}\n```\n\n"
        f"Quant Analyst signal:\n```json\n{quant_json}\n```\n\n"
        f"Devil's Advocate critique:\n```json\n{critique_json}\n```\n\n"
        f"Risk Manager assessment:\n```json\n{risk_json}\n```\n\n"
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
        "  MUST be 0.0, BUT the 'confidence' score must be calculated dynamically based on agent "
        "  consensus. If all agents strongly agree the market is untradable, output high confidence "
        "  (0.80-0.99). If there is mixed sentiment but HOLD wins out of caution, output lower "
        "  confidence (0.50-0.70). DO NOT hardcode a specific number and DO NOT output confidence = 0.0 "
        "  for a HOLD decision.\n"
        "- Confidence reflects your conviction after weighing Quant + Devil's Advocate + Risk Manager\n"
        "- Reasoning should be 1-2 sentences max\n"
        f"- The final 'reasoning' field in the JSON MUST be written in {market.language} "
        "(if 'vi' use Vietnamese, if 'en' use English). All other keys and processes remain in English."
    )

    agent = _execution_judge_agent(_build_reasoner_llm())
    task = Task(
        description=prompt,
        expected_output="A single valid JSON object matching the AiCouncilDecision schema.",
        agent=agent,
    )
    raw = agent.execute_task(task)

    try:
        final_decision = _parse_agent_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Execution Judge output parse failed: %s | raw: %s", exc, raw[:500])
        final_decision = {
            "action": "HOLD",
            "confidence": 0.0,
            "entry": 0.0,
            "entry_condition": "",
            "leverage": 0,
            "position_size_pct": 0.0,
            "stop_loss": 0.0,
            "invalidation_point": "",
            "take_profit_1": 0.0,
            "take_profit_2": 0.0,
            "reasoning": f"Execution Judge parse failed: {exc}",
        }

    return {"final_decision": final_decision}


# ---------------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------------

def route_after_devil(state: AiCouncilState) -> str:
    """Route back to Quant Analyst for re-evaluation if severe contradiction found.

    Prevents infinite loops via MAX_DEBATE_ITERATIONS.
    """
    devil_critique = state.get("devil_critique", {})
    iteration_count = state.get("iteration_count", 0)

    verdict = devil_critique.get("verdict", "")
    severe = verdict == "strong_contradiction"

    if severe and iteration_count < MAX_DEBATE_ITERATIONS:
        logger.info(
            "Devil's Advocate found strong contradictions (iteration %d/%d); re-routing to Quant Analyst",
            iteration_count,
            MAX_DEBATE_ITERATIONS,
        )
        _emit_info(
            state,
            f"🔄 Severe contradictions detected — routing back to Quant Analyst (Iteration {iteration_count + 1}/{MAX_DEBATE_ITERATIONS})",
        )
        return "quant_analyst"

    if severe:
        _emit_info(
            state,
            f"⚠️ Severe contradictions remain but max debate iterations ({MAX_DEBATE_ITERATIONS}) reached — proceeding to Risk Manager",
        )
    else:
        _emit_info(state, "✅ Devil's Advocate critique complete — proceeding to Risk Manager")

    return "risk_manager"


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------

def build_ai_council_graph() -> StateGraph:
    """Build and return the compiled LangGraph AI Council debate graph."""
    workflow = StateGraph(AiCouncilState)

    workflow.add_node("quant_analyst", quant_analyst_node)
    workflow.add_node("devils_advocate", devils_advocate_node)
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("execution_judge", execution_judge_node)

    workflow.set_entry_point("quant_analyst")

    workflow.add_edge("quant_analyst", "devils_advocate")
    workflow.add_conditional_edges(
        "devils_advocate",
        route_after_devil,
        {
            "quant_analyst": "quant_analyst",
            "risk_manager": "risk_manager",
        },
    )
    workflow.add_edge("risk_manager", "execution_judge")
    workflow.add_edge("execution_judge", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ai_council_graph(market: MarketContext) -> AiCouncilDecision:
    """Run the full cyclic AI Council and return a structured decision."""
    graph = build_ai_council_graph()

    initial_state: AiCouncilState = {
        "market_data": market.model_dump(mode="json"),
        "quant_signal": {},
        "devil_critique": {},
        "risk_assessment": {},
        "final_decision": {},
        "iteration_count": 0,
        "event_queue": None,
    }

    final_state = graph.invoke(initial_state)
    return AiCouncilDecision(**final_state["final_decision"])


# Maps each graph node name to the terminal label used by the frontend
# (AiCouncilPanel's AGENT_COLORS) and the state key holding its parsed JSON output.
_NODE_LABELS: dict[str, tuple[str, str]] = {
    "quant_analyst": ("📊 Quant Analyst", "quant_signal"),
    "devils_advocate": ("😈 Devil's Advocate", "devil_critique"),
    "risk_manager": ("🛡️ Risk Manager", "risk_assessment"),
    "execution_judge": ("⚖️ Execution Judge", "final_decision"),
}


def run_ai_council_graph_streaming(
    market: MarketContext,
) -> Generator[str, None, None]:
    """Run the cyclic AI Council in a background thread and yield SSE events.

    Yields the same SSE protocol as the legacy CrewAI streaming function:
        data: [CONNECTED]\n\n
        data: [INFO] ...\n\n
        data: [FINAL_RESULT]:<compact AiCouncilDecision JSON>\n\n
        data: [ERROR]:...\n\n
    """
    event_queue: Queue[str | object] = Queue()
    sentinel = object()

    def _run_council() -> None:
        """Background worker: runs the blocking LangGraph pipeline.

        Uses graph.stream(..., stream_mode="updates") instead of invoke() so that
        each node's parsed JSON output can be pushed to the event queue as soon as
        that node completes, giving the frontend terminal the full agent payloads
        (not just short [INFO] narration) in real time.
        """
        try:
            graph = build_ai_council_graph()
            initial_state: AiCouncilState = {
                "market_data": market.model_dump(mode="json"),
                "quant_signal": {},
                "devil_critique": {},
                "risk_assessment": {},
                "final_decision": {},
                "iteration_count": 0,
                "event_queue": event_queue,
            }

            accumulated_state: dict[str, Any] = dict(initial_state)
            for step in graph.stream(initial_state, stream_mode="updates"):
                for node_name, update in step.items():
                    accumulated_state.update(update)

                    label_and_key = _NODE_LABELS.get(node_name)
                    if label_and_key is None:
                        continue
                    label, state_key = label_and_key
                    payload = update.get(state_key)
                    if not payload:
                        continue
                    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
                    event_queue.put(f"[{label}] completed analysis:\n{payload_json}")

            decision = AiCouncilDecision(**accumulated_state["final_decision"])
            compact_json = json.dumps(decision.model_dump(), separators=(",", ":"))
            event_queue.put(f"[FINAL_RESULT]:{compact_json}")
        except Exception as exc:
            logger.exception("LangGraph council failed")
            error_tb = traceback.format_exc().replace("\n", " | ")
            event_queue.put(f"[ERROR]:LangGraph council failed - {type(exc).__name__}: {exc}")
            event_queue.put(f"[TRACE]:{error_tb}")
        finally:
            event_queue.put(sentinel)

    # Flush headers immediately so the browser doesn't hang on provisional headers.
    yield "data: [CONNECTED]\n\n"

    thread = threading.Thread(target=_run_council, daemon=True)
    thread.start()

    try:
        while True:
            try:
                item = event_queue.get(timeout=15)
            except Empty:
                yield "data: [KEEPALIVE]\n\n"
                continue

            if item is sentinel:
                break

            message = str(item)

            # Protocol messages must be emitted as a single data: line.
            if (
                message.startswith("[FINAL_RESULT]:")
                or message.startswith("[ERROR]:")
                or message.startswith("[TRACE]:")
                or message.startswith("[INFO]")
            ):
                yield f"data: {message}\n\n"
                continue

            # Regular multi-line agent output — split per SSE spec.
            for line in message.split("\n"):
                yield f"data: {line}\n"
            yield "\n"
    except Exception as gen_exc:
        logger.exception("SSE generator crash")
        error_tb = traceback.format_exc().replace("\n", " | ")
        yield f"data: [ERROR]:SSE Stream Error - {type(gen_exc).__name__}: {gen_exc}\n\n"
        yield f"data: [TRACE]:{error_tb}\n\n"
