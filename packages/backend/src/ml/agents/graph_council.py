"""LangGraph-based iterative debate council for AetherForecast.

The graph forces the Quant Analyst's proposal through adversarial scrutiny by
a Devil's Advocate before the Risk Manager and Execution Judge see it. If the
Advocate issues a "strong_contradiction" or "weak_signal" verdict, the graph
loops back to the Quant Analyst for revision, making the system highly
skeptical by design.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.ml.agents.crew import (
    AiCouncilDecision,
    MarketContext,
    TradeAction,
    _build_chat_llm,
    _build_reasoner_llm,
    _execution_judge_agent,
    _quant_analyst_agent,
    _risk_manager_agent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class CouncilState(TypedDict):
    """Shared state passed between graph nodes."""

    market: MarketContext
    market_json: str
    quant_proposal: str
    devil_verdict: str
    risk_assessment: str
    final_decision: AiCouncilDecision | None
    revision_count: int
    logs: list[str]


_MAX_REVISIONS = 2


# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------


def _devils_advocate_agent(llm: Any) -> Any:
    """Adversarial agent that challenges the Quant Analyst's bias."""
    return __import__("crewai", fromlist=["Agent"]).Agent(
        role="Devil's Advocate",
        goal=(
            "Ruthlessly challenge the Quant Analyst's trade proposal. Identify confirmation bias, "
            "overfitting to recent candles, ignored tail risks, and contradictory technical evidence. "
            "You are not trying to be right; you are trying to raise the standard of evidence."
        ),
        backstory=(
            "You are a skeptical senior quant at a hedge fund who has seen many "
            "false breakouts and biased models. Your job is to play devil's advocate "
            "and force the team to either strengthen the signal or walk away."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------


def _run_agent(agent: Any, task_description: str) -> str:
    """Execute a single CrewAI agent against a task description."""
    Task = __import__("crewai", fromlist=["Task"]).Task
    task = Task(description=task_description, expected_output="Concise analysis.", agent=agent)
    return agent.execute_task(task)


def quant_analyst_node(state: CouncilState) -> CouncilState:
    """Produce (or revise) the Quant Analyst's trade proposal."""
    market = state["market"]
    market_json = state["market_json"]
    is_revision = bool(state.get("devil_verdict"))

    prompt = (
        f"Analyze the following market data and Chronos-2 forecast (Risk Profile: {market.risk_profile.value}):\n\n"
        f"```json\n{market_json}\n```\n\n"
    )
    if is_revision:
        prompt += (
            "The Devil's Advocate has challenged your previous proposal. "
            f"Address the critique below and revise your proposal accordingly:\n\n"
            f"```\n{state['devil_verdict']}\n```\n\n"
            "If the critique is valid, either neutralize your bias or switch to NEUTRAL/HOLD. "
            "If you still believe in the signal, provide stronger evidence.\n\n"
        )
    prompt += (
        "Provide:\n"
        "1. Directional bias: LONG, SHORT, or NEUTRAL\n"
        "2. Confidence score (0-1)\n"
        "3. Suggested entry price\n"
        "4. Entry condition\n"
        "5. Initial stop-loss price\n"
        "6. Two take-profit levels\n"
        "7. Invalidation point\n"
        "8. Brief technical reasoning (2-3 sentences)\n\n"
        "Consider forecast bands, current price vs. forecast median, realized volatility, "
        "funding rate, and fear/greed index."
    )

    llm = _build_chat_llm()
    agent = _quant_analyst_agent(llm)
    result = _run_agent(agent, prompt)

    return {
        **state,
        "quant_proposal": result,
        "revision_count": state.get("revision_count", 0) + (1 if is_revision else 0),
        "logs": [
            *state["logs"],
            "📊 Quant Analyst issued a proposal." if not is_revision else "📊 Quant Analyst revised the proposal.",
            result,
        ],
    }


def devil_advocate_node(state: CouncilState) -> CouncilState:
    """Challenge the Quant Analyst's proposal."""
    market_json = state["market_json"]
    proposal = state["quant_proposal"]

    prompt = (
        "You are the Devil's Advocate. Review the Quant Analyst's proposal below and the market context.\n\n"
        f"Market context:\n```json\n{market_json}\n```\n\n"
        f"Quant Analyst proposal:\n```\n{proposal}\n```\n\n"
        "Output a single verdict line at the end of your response in this exact format:\n"
        "verdict: strong_contradiction | weak_signal | acceptable\n\n"
        "Definitions:\n"
        "- strong_contradiction: the proposal contradicts clear technical evidence or ignores major risk.\n"
        "- weak_signal: the evidence is insufficient, the setup is marginal, or the risk/reward is poor.\n"
        "- acceptable: the proposal is well-supported and the risk/reward is reasonable.\n\n"
        "Be skeptical. If in doubt, choose weak_signal."
    )

    llm = _build_chat_llm()
    agent = _devils_advocate_agent(llm)
    result = _run_agent(agent, prompt)

    return {
        **state,
        "devil_verdict": result,
        "logs": [
            *state["logs"],
            "🎭 Devil's Advocate issued a verdict.",
            result,
        ],
    }


def risk_manager_node(state: CouncilState) -> CouncilState:
    """Review the proposal from a risk perspective."""
    market = state["market"]
    market_json = state["market_json"]
    proposal = state["quant_proposal"]

    prompt = (
        "Review the Quant Analyst's trade proposal against risk factors. "
        f"The trader's risk profile is {market.risk_profile.value}.\n\n"
        f"Market context:\n```json\n{market_json}\n```\n\n"
        f"Quant Analyst proposal:\n```\n{proposal}\n```\n\n"
        "Evaluate:\n"
        "1. Sentiment score: extreme values (> 0.7 or < -0.7) signal caution\n"
        "2. Funding rate: high positive = crowded longs, high negative = crowded shorts\n"
        "3. Fear/Greed Index: extreme fear (< 20) or extreme greed (> 80) = caution\n"
        "4. Realized volatility: high vol = reduce leverage / position size\n\n"
        "STRICT LEVERAGE BOUNDS:\n"
        "- CONSERVATIVE: leverage must be 2x-10x\n"
        "- BALANCED: leverage must be 11x-40x\n"
        "- DEGEN: leverage must be 41x-125x\n\n"
        "Output: approved/reduced/vetoed, adjusted leverage (respect bounds), "
        "adjusted stop-loss, position_size_pct, invalidation_point, take_profit_1, take_profit_2, reasoning"
    )

    llm = _build_chat_llm()
    agent = _risk_manager_agent(llm, market.risk_profile)
    result = _run_agent(agent, prompt)

    return {
        **state,
        "risk_assessment": result,
        "logs": [
            *state["logs"],
            "🛡️ Risk Manager completed risk assessment.",
            result,
        ],
    }


def execution_judge_node(state: CouncilState) -> CouncilState:
    """Synthesize everything into a final AiCouncilDecision."""
    market = state["market"]
    proposal = state["quant_proposal"]
    risk = state["risk_assessment"]

    prompt = (
        "You have the Quant Analyst's proposal and the Risk Manager's assessment. "
        f"The trader's risk profile is {market.risk_profile.value}. "
        "Make the final pro-trader trade decision.\n\n"
        f"Quant Analyst proposal:\n```\n{proposal}\n```\n\n"
        f"Risk Manager assessment:\n```\n{risk}\n```\n\n"
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
        f'  "reasoning": "<concise string in {market.language}>"\n'
        "}\n\n"
        "Rules:\n"
        "- If Risk Manager vetoed, action MUST be HOLD with entry/SL/TPs = 0\n"
        "- If reduced, apply the Risk Manager's leverage and stop-loss adjustments\n"
        "- Leverage MUST respect the risk profile bounds\n"
        "- Confidence reflects your conviction after both analyses\n"
        "- Reasoning should be 1-2 sentences max\n"
        f"- The final 'reasoning' field in the JSON MUST be written in {market.language} "
        "(if 'vi' use Vietnamese, if 'en' use English). All other keys and processes remain in English."
    )

    llm = _build_reasoner_llm()
    agent = _execution_judge_agent(llm)
    raw = _run_agent(agent, prompt)
    decision = _parse_trade_decision(raw)

    return {
        **state,
        "final_decision": decision,
        "logs": [
            *state["logs"],
            "⚖️ Execution Judge rendered the final decision.",
            raw,
        ],
    }


# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------


def route_after_devil(state: CouncilState) -> str:
    """Decide whether to send the proposal back for revision or to risk review.

    The system is intentionally skeptical: both 'strong_contradiction' and
    'weak_signal' verdicts trigger a revision loop. Only 'acceptable' moves
    forward. A hard cap on revisions prevents infinite loops.
    """
    verdict = state["devil_verdict"].lower()
    if state["revision_count"] >= _MAX_REVISIONS:
        logger.info("Max revisions (%d) reached; moving to Risk Manager", _MAX_REVISIONS)
        return "risk_manager"
    if "strong_contradiction" in verdict or "weak_signal" in verdict:
        logger.info(
            "Devil's Advocate returned skeptical verdict (%s); routing back to Quant Analyst (revision %d/%d)",
            verdict[:80],
            state["revision_count"] + 1,
            _MAX_REVISIONS,
        )
        return "quant_analyst"
    return "risk_manager"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_council_graph() -> StateGraph:
    """Build and return the compiled LangGraph debate graph."""
    workflow = StateGraph(CouncilState)

    workflow.add_node("quant_analyst", quant_analyst_node)
    workflow.add_node("devil_advocate", devil_advocate_node)
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("execution_judge", execution_judge_node)

    workflow.set_entry_point("quant_analyst")
    workflow.add_edge("quant_analyst", "devil_advocate")
    workflow.add_conditional_edges(
        "devil_advocate",
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
# Output Parsing
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
            leverage=1,
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


def run_graph_council_streaming(
    market: MarketContext,
) -> Generator[str, None, None]:
    """Execute the graph council and yield SSE events as agents work.

    Yields SSE-formatted strings. The final event contains
    `[FINAL_RESULT]:<json>` for the AiCouncilDecision.
    """
    graph = build_council_graph()
    initial_state: CouncilState = {
        "market": market,
        "market_json": market.model_dump_json(indent=2),
        "quant_proposal": "",
        "devil_verdict": "",
        "risk_assessment": "",
        "final_decision": None,
        "revision_count": 0,
        "logs": ["🚀 AI Council debate session started."],
    }

    last_log_count = 0
    for state in graph.stream(initial_state, stream_mode="values"):
        logs = state.get("logs", [])
        while last_log_count < len(logs):
            yield f"data: {logs[last_log_count].replace(chr(10), ' | ')}\n\n"
            last_log_count += 1

    final_decision = state.get("final_decision")
    if final_decision is not None:
        yield f"data: [FINAL_RESULT]:{final_decision.model_dump_json()}\n\n"
    else:
        yield f"data: [FINAL_RESULT]:{_parse_trade_decision('').model_dump_json()}\n\n"
