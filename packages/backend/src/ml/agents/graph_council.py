"""LangGraph-based iterative debate council for AetherForecast.

The graph runs the Quant Analyst's proposal through adversarial scrutiny by a
Devil's Advocate, but the debate is two-way: the Advocate approves
well-supported setups (recommending a lower position size / tighter stop
instead of a reflexive veto) rather than defaulting to HOLD, and the Quant
Analyst can rebut a critique with added justification instead of caving to it
outright. A strict retry cap (_MAX_RETRIES) prevents infinite back-and-forth;
if the two haven't converged after that many rounds, the full debate
transcript is carried forward to the Execution Judge, who acts as the final
arbitrator and makes the definitive LONG/SHORT/HOLD call.
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
    # Full Quant Analyst <-> Devil's Advocate exchange across every round, kept
    # separate from `logs` (which also carries Risk Manager / Judge output and
    # UI-facing system messages) so the Execution Judge can be handed exactly
    # the debate transcript when it acts as final arbitrator.
    debate_transcript: list[str]


_MAX_RETRIES = 2


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
            f"Address the critique below:\n\n"
            f"```\n{state['devil_verdict']}\n```\n\n"
            "This is a two-way debate, not an automatic override: if the Devil's Advocate's "
            "critique is valid, adjust the proposal accordingly (revise bias, entry, stop-loss, "
            "or position sizing as needed). HOWEVER, if your initial technical data (e.g. strong "
            "volume, clear trend) is robust, REBUT the critique directly and maintain your "
            "directional bias with added justification — do not cave to skepticism just because "
            "it was raised.\n\n"
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
    # Use the post-increment revision_count so this round's label matches the
    # Devil's Advocate round that follows it (which reads revision_count as-is).
    new_revision_count = state.get("revision_count", 0) + (1 if is_revision else 0)
    round_label = f"Round {new_revision_count + 1}"

    return {
        **state,
        "quant_proposal": result,
        "revision_count": new_revision_count,
        "logs": [
            *state["logs"],
            "📊 Quant Analyst issued a proposal." if not is_revision else "📊 Quant Analyst revised the proposal.",
            result,
        ],
        "debate_transcript": [
            *state.get("debate_transcript", []),
            f"[{round_label}] Quant Analyst: {result}",
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
        "Your job is to raise the standard of evidence, NOT to reflexively veto or force a HOLD by "
        "default. Judge the proposal on its actual merits:\n"
        "- If the setup has a Risk/Reward ratio better than 1:2 AND is backed by strong technicals "
        "(e.g. volume spikes, RSI divergence, clear trend structure), APPROVE it. Do not block a "
        "well-supported trade just to be cautious — instead, explicitly recommend a LOW POSITION "
        "SIZE and a TIGHT STOP-LOSS in your reasoning so the trade proceeds with reduced risk rather "
        "than being vetoed outright.\n"
        "- Reserve a harsher verdict for setups that actually deserve it: confirmation bias, "
        "overfitting to recent candles, ignored tail risks, or genuinely contradictory technical "
        "evidence — not merely \"could be safer.\"\n\n"
        "Output a single verdict line at the end of your response in this exact format:\n"
        "verdict: strong_contradiction | weak_signal | acceptable\n\n"
        "Definitions:\n"
        "- strong_contradiction: the proposal contradicts clear technical evidence or ignores a major "
        "risk that reducing position size / tightening the stop would not address.\n"
        "- weak_signal: the risk/reward is genuinely poor (worse than 1:2) or the evidence is "
        "insufficient, independent of position sizing.\n"
        "- acceptable: the proposal is well-supported, OR has good risk/reward and strong technicals — "
        "including cases where you're approving with a recommended lower position size and tighter "
        "stop instead of a flat veto.\n\n"
        "CRITICAL LANGUAGE RULE: You MUST output your reasoning, analysis, and all values strictly in "
        "ENGLISH. Do NOT translate professional trading terminology (e.g., Long, Short, Breakout, "
        "Stop-Loss, Take-Profit, Volatility, Range-bound) into Vietnamese or any other language.\n\n"
        "Be rigorous, not reflexively skeptical — a well-supported trade with sensible risk controls "
        "should be approved, not held hostage by default caution."
    )

    llm = _build_chat_llm()
    agent = _devils_advocate_agent(llm)
    result = _run_agent(agent, prompt)
    round_label = f"Round {state.get('revision_count', 0) + 1}"

    return {
        **state,
        "devil_verdict": result,
        "logs": [
            *state["logs"],
            "🎭 Devil's Advocate issued a verdict.",
            result,
        ],
        "debate_transcript": [
            *state.get("debate_transcript", []),
            f"[{round_label}] Devil's Advocate: {result}",
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
        "CRITICAL LANGUAGE RULE: You MUST output your reasoning, analysis, and all values strictly in "
        "ENGLISH. Do NOT translate professional trading terminology (e.g., Long, Short, Breakout, "
        "Stop-Loss, Take-Profit, Volatility, Range-bound) into Vietnamese or any other language.\n\n"
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
    risk = state["risk_assessment"]
    debate_transcript = state.get("debate_transcript", [])
    debate_log = "\n\n".join(debate_transcript) if debate_transcript else "(no debate rounds recorded)"

    prompt = (
        "You are the final arbitrator. Below is the FULL debate transcript between the Quant "
        "Analyst and the Devil's Advocate across every round (up to the retry cap), followed by "
        "the Risk Manager's assessment. Base your LONG/SHORT/HOLD call primarily on this debate "
        "log — whether the Quant Analyst's rebuttals held up under scrutiny, or whether the "
        "Devil's Advocate's critiques went unanswered.\n\n"
        f"The trader's risk profile is {market.risk_profile.value}.\n\n"
        f"Full debate transcript:\n```\n{debate_log}\n```\n\n"
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
        '  "reasoning": "<concise string in the requested language>"\n'
        "}\n\n"
        "Rules:\n"
        "- If Risk Manager vetoed, action MUST be HOLD with entry/SL/TPs = 0\n"
        "- If reduced, apply the Risk Manager's leverage and stop-loss adjustments\n"
        "- Leverage MUST respect the risk profile bounds\n"
        "- Your action MUST be a definitive call grounded in the debate transcript — do not default "
        "  to HOLD merely because the debate was contentious. If the Quant Analyst's final rebuttal "
        "  held up against the Devil's Advocate's critique, honor the directional bias; if the "
        "  critique went unanswered, side with the Devil's Advocate.\n"
        "- Confidence reflects your conviction after weighing the full debate and the Risk Manager's "
        "  assessment\n"
        "- Reasoning should be 1-2 sentences max\n"
        f"- CRITICAL LOCALIZATION RULE: The user has requested the output in the '{market.language}' "
        "language ('vi' for Vietnamese, 'en' for English). You MUST write the 'reasoning' and "
        "'entry_condition' fields in this requested language. However, to maintain professional "
        "financial tone, you MUST NOT translate trading terminology. Keep words like LONG, SHORT, "
        "Take-Profit, Stop-Loss, Breakout, Limit Order, Market Order, Volatility strictly in "
        "English, even if the rest of the sentence is in the requested language."
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

    Only a genuine 'strong_contradiction' or 'weak_signal' verdict triggers a
    revision loop — the Devil's Advocate is instructed to approve
    well-supported setups (with reduced size / tighter stop) rather than
    reflexively raising one of these, so this is no longer "skeptical by
    default." A strict retry cap (_MAX_RETRIES) prevents infinite
    back-and-forth: once hit, the debate breaks and the full transcript moves
    on to the Risk Manager and then the Execution Judge, who arbitrates.
    """
    verdict = state["devil_verdict"].lower()
    if state["revision_count"] >= _MAX_RETRIES:
        logger.info(
            "Max retries (%d) reached without consensus; breaking the debate loop and handing the "
            "transcript to Risk Manager -> Execution Judge",
            _MAX_RETRIES,
        )
        return "risk_manager"
    if "strong_contradiction" in verdict or "weak_signal" in verdict:
        logger.info(
            "Devil's Advocate returned a skeptical verdict (%s); routing back to Quant Analyst for "
            "rebuttal (retry %d/%d)",
            verdict[:80],
            state["revision_count"] + 1,
            _MAX_RETRIES,
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
        "debate_transcript": [],
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
