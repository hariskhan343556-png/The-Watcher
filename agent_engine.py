"""
agent_engine.py
----------------
Agentic decision engine for "The Watcher".

Pipeline (4 nodes):
    1. ingest_node    - normalizes the incoming activity stream
    2. score_node     - computes Toxicity / Addiction / Safety scores
                         (heuristics + a simple scikit-learn trend model)
    3. reason_node     - an LLM reasoning step (OpenAI or local Ollama via
                         LangChain) that explains *why* the scores look the
                         way they do. Falls back to deterministic templated
                         reasoning if no LLM backend is configured.
    4. action_node    - turns scores + reasoning into concrete alerts and
                         a parent-facing recommendation list.

The pipeline is expressed as a LangGraph StateGraph when the `langgraph`
package is installed; otherwise a tiny sequential fallback runner with the
identical node functions is used, so the app works fully offline with zero
extra dependencies beyond pandas/numpy/scikit-learn.
"""

import os
import datetime as dt
from typing import TypedDict, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from data_simulator import CONCERNING_PHRASES

# --------------------------------------------------------------------------
# Optional dependency detection (the app must work with none of these
# installed / no API key set -- everything degrades gracefully).
# --------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    import langchain_openai  # noqa: F401
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

try:
    import langchain_ollama  # noqa: F401
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False


DEFAULT_RULES = {
    "max_screen_time_hours": 4.0,
    "bedtime_start": dt.time(22, 0),
    "bedtime_end": dt.time(6, 0),
    "blocked_categories_at_night": ["Games", "Social Media", "Video Streaming"],
    "toxicity_alert_threshold": 40,   # 0-100, higher = more sensitive
    "addiction_alert_threshold": 50,  # 0-100, higher = more sensitive
}


# --------------------------------------------------------------------------
# State definition
# --------------------------------------------------------------------------
class WatcherState(TypedDict, total=False):
    rules: Dict[str, Any]
    screen_time_df: pd.DataFrame
    app_usage_df: pd.DataFrame
    messages_df: pd.DataFrame
    activity_log_df: pd.DataFrame

    toxicity_score: float
    addiction_score: float
    safety_score: float
    risk_category: str

    reasoning_text: str
    llm_backend_used: str

    alerts: List[Dict[str, Any]]
    report: Dict[str, Any]


# --------------------------------------------------------------------------
# Scoring heuristics
# --------------------------------------------------------------------------
def calculate_toxicity_score(messages_df: pd.DataFrame) -> float:
    """0 (safe) - 100 (severe) based on flagged messages and sentiment."""
    if messages_df is None or messages_df.empty:
        return 0.0

    flagged = messages_df.get("concern_flag")
    flag_ratio = float(flagged.mean()) if flagged is not None and len(flagged) else 0.0

    avg_sentiment = float(messages_df["sentiment_score"].mean()) if "sentiment_score" in messages_df else 0.0
    # sentiment in [-1, 1]; negative sentiment increases score
    sentiment_component = max(0.0, -avg_sentiment) * 50

    keyword_hits = messages_df["message_snippet"].fillna("").str.lower().apply(
        lambda t: any(p in t for p in CONCERNING_PHRASES)
    ).sum() if "message_snippet" in messages_df else 0

    score = flag_ratio * 60 + sentiment_component + min(keyword_hits * 5, 20)
    return round(float(np.clip(score, 0, 100)), 1)


def calculate_addiction_score(screen_time_df: pd.DataFrame, activity_log_df: pd.DataFrame, rules: Dict[str, Any]) -> float:
    """0 (healthy) - 100 (high risk) based on daily average, trend slope and late-night use."""
    if screen_time_df is None or screen_time_df.empty:
        return 0.0

    threshold = rules.get("max_screen_time_hours", DEFAULT_RULES["max_screen_time_hours"])
    avg_hours = float(screen_time_df["screen_time_hours"].mean())
    over_threshold_component = max(0.0, (avg_hours - threshold) / threshold) * 45
    over_threshold_component = min(over_threshold_component, 45)

    # Trend: is daily screen time escalating? Simple linear regression slope.
    x = np.arange(len(screen_time_df)).reshape(-1, 1)
    y = screen_time_df["screen_time_hours"].values
    slope = 0.0
    if len(x) >= 3:
        model = LinearRegression().fit(x, y)
        slope = float(model.coef_[0])
    trend_component = min(max(slope, 0) * 25, 30)

    # Late-night usage ratio from the raw activity log
    late_ratio = 0.0
    if activity_log_df is not None and not activity_log_df.empty and "late_night" in activity_log_df:
        late_ratio = float(activity_log_df["late_night"].mean())
    late_component = late_ratio * 25

    score = over_threshold_component + trend_component + late_component
    return round(float(np.clip(score, 0, 100)), 1)


def compute_safety_score(toxicity_score: float, addiction_score: float) -> float:
    """Overall Safety Score (0-100, higher = safer)."""
    harm = 0.55 * toxicity_score + 0.45 * addiction_score
    return round(float(np.clip(100 - harm, 0, 100)), 1)


def classify_risk_category(safety_score: float, toxicity_score: float, addiction_score: float) -> str:
    if toxicity_score >= 40:
        return "Toxic Content Exposure"
    if addiction_score >= 50:
        return "Mild Addiction Risk" if addiction_score < 75 else "Severe Addiction Risk"
    if safety_score < 60:
        return "Elevated Risk"
    return "Safe"


# --------------------------------------------------------------------------
# LLM reasoning (with graceful fallback)
# --------------------------------------------------------------------------
def _build_reasoning_prompt(context: Dict[str, Any]) -> str:
    return f"""You are a careful, evidence-based child digital-safety analyst helping a parent
understand their child's recent device activity. Do not be alarmist; be specific and practical.

Data summary:
- Average daily screen time: {context['avg_screen_time']:.1f} hours (family limit: {context['max_screen_time']:.1f} hours)
- Late-night usage ratio: {context['late_ratio']*100:.0f}% of sessions occurred between 10pm-6am
- Toxicity score: {context['toxicity_score']}/100
- Addiction score: {context['addiction_score']}/100
- Safety score: {context['safety_score']}/100
- Number of flagged concerning messages: {context['flagged_count']}
- Dominant app category: {context['dominant_category']}

In 3-5 sentences, explain what pattern this data suggests, how concerned a parent should
reasonably be, and one concrete, age-appropriate next step. Avoid diagnosing the child;
describe behavior patterns only."""


def _heuristic_reasoning(context: Dict[str, Any]) -> str:
    parts = []
    if context["toxicity_score"] >= 40:
        parts.append(
            f"Several messages ({context['flagged_count']}) contain language patterns "
            "consistent with bullying or hostility, and overall message sentiment is negative. "
            "This warrants a calm, direct conversation with the child about what they're "
            "experiencing online."
        )
    if context["addiction_score"] >= 50:
        parts.append(
            f"Average screen time ({context['avg_screen_time']:.1f}h/day) is above the configured "
            f"limit of {context['max_screen_time']:.1f}h, with {context['late_ratio']*100:.0f}% of "
            f"sessions occurring late at night, concentrated in {context['dominant_category']}. "
            "This pattern is consistent with compulsive/addictive use rather than incidental overuse."
        )
    if not parts:
        parts.append(
            "Activity levels and message sentiment are within a healthy range for this period. "
            "No specific intervention appears necessary; continue routine monitoring."
        )
    return " ".join(parts)


def llm_reason(context: Dict[str, Any]) -> Dict[str, str]:
    """Returns {'text': ..., 'backend': ...}. Tries OpenAI, then Ollama, then heuristic."""
    api_key = os.environ.get("OPENAI_API_KEY")
    use_ollama = os.environ.get("WATCHER_USE_OLLAMA", "false").lower() == "true"
    prompt = _build_reasoning_prompt(context)

    if api_key and LANGCHAIN_OPENAI_AVAILABLE:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.environ.get("WATCHER_OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
            resp = llm.invoke(prompt)
            return {"text": resp.content.strip(), "backend": "openai"}
        except Exception as e:
            return {"text": _heuristic_reasoning(context) + f"\n\n(LLM call failed, used rule-based fallback: {e})",
                    "backend": "heuristic-fallback"}

    if use_ollama and LANGCHAIN_OLLAMA_AVAILABLE:
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=os.environ.get("WATCHER_OLLAMA_MODEL", "llama3"))
            resp = llm.invoke(prompt)
            return {"text": resp.content.strip(), "backend": "ollama"}
        except Exception as e:
            return {"text": _heuristic_reasoning(context) + f"\n\n(Ollama call failed, used rule-based fallback: {e})",
                    "backend": "heuristic-fallback"}

    return {"text": _heuristic_reasoning(context), "backend": "heuristic"}


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------
def node_ingest(state: WatcherState) -> WatcherState:
    # Normalize / defend against missing frames
    for key in ["screen_time_df", "app_usage_df", "messages_df", "activity_log_df"]:
        if state.get(key) is None:
            state[key] = pd.DataFrame()
    return state


def node_score(state: WatcherState) -> WatcherState:
    rules = state.get("rules", DEFAULT_RULES)
    toxicity = calculate_toxicity_score(state["messages_df"])
    addiction = calculate_addiction_score(state["screen_time_df"], state["activity_log_df"], rules)
    safety = compute_safety_score(toxicity, addiction)
    risk = classify_risk_category(safety, toxicity, addiction)

    state["toxicity_score"] = toxicity
    state["addiction_score"] = addiction
    state["safety_score"] = safety
    state["risk_category"] = risk
    return state


def node_reason(state: WatcherState) -> WatcherState:
    messages_df = state["messages_df"]
    activity_log_df = state["activity_log_df"]
    rules = state.get("rules", DEFAULT_RULES)

    flagged_count = int(messages_df["concern_flag"].sum()) if "concern_flag" in messages_df else 0
    late_ratio = float(activity_log_df["late_night"].mean()) if "late_night" in activity_log_df and not activity_log_df.empty else 0.0
    dominant_category = (
        activity_log_df["category"].mode().iloc[0]
        if "category" in activity_log_df and not activity_log_df.empty
        else "N/A"
    )
    avg_screen_time = float(state["screen_time_df"]["screen_time_hours"].mean()) if not state["screen_time_df"].empty else 0.0

    context = {
        "avg_screen_time": avg_screen_time,
        "max_screen_time": rules.get("max_screen_time_hours", DEFAULT_RULES["max_screen_time_hours"]),
        "late_ratio": late_ratio,
        "toxicity_score": state["toxicity_score"],
        "addiction_score": state["addiction_score"],
        "safety_score": state["safety_score"],
        "flagged_count": flagged_count,
        "dominant_category": dominant_category,
    }

    result = llm_reason(context)
    state["reasoning_text"] = result["text"]
    state["llm_backend_used"] = result["backend"]
    return state


def node_action(state: WatcherState) -> WatcherState:
    rules = state.get("rules", DEFAULT_RULES)
    alerts = []
    now = dt.datetime.now()

    if state["toxicity_score"] >= rules.get("toxicity_alert_threshold", 40):
        severity = "Critical" if state["toxicity_score"] >= 70 else "Medium"
        alerts.append({
            "timestamp": now, "severity": severity, "category": "Toxic Content Exposure",
            "message": f"Toxicity score {state['toxicity_score']}/100 — concerning language detected in messages. "
                       "Recommended action: review flagged conversations and talk with your child.",
        })

    if state["addiction_score"] >= rules.get("addiction_alert_threshold", 50):
        severity = "Critical" if state["addiction_score"] >= 75 else "Medium"
        alerts.append({
            "timestamp": now, "severity": severity, "category": "Addiction Risk",
            "message": f"Addiction score {state['addiction_score']}/100 — escalating/above-limit screen time detected. "
                       "Recommended action: enable an app-lock rule after bedtime and set a daily cap.",
        })

    activity_log_df = state["activity_log_df"]
    if not activity_log_df.empty and "late_night" in activity_log_df:
        night_blocked = activity_log_df[
            activity_log_df["late_night"] & activity_log_df["category"].isin(rules.get("blocked_categories_at_night", []))
        ]
        if len(night_blocked) > 0:
            alerts.append({
                "timestamp": now, "severity": "Low", "category": "Bedtime Rule Violation",
                "message": f"{len(night_blocked)} session(s) in blocked categories occurred during bedtime hours. "
                           "Recommended action: enforce auto-lock for these categories after bedtime.",
            })

    if not alerts:
        alerts.append({
            "timestamp": now, "severity": "Low", "category": "Safe",
            "message": "No rule thresholds were exceeded in this period. Continue routine monitoring.",
        })

    state["alerts"] = alerts
    state["report"] = {
        "generated_at": now,
        "risk_category": state["risk_category"],
        "safety_score": state["safety_score"],
        "toxicity_score": state["toxicity_score"],
        "addiction_score": state["addiction_score"],
        "reasoning_text": state["reasoning_text"],
        "llm_backend_used": state["llm_backend_used"],
        "alerts": alerts,
    }
    return state


NODE_SEQUENCE = [
    ("Ingest Activity Stream", node_ingest),
    ("Calculate Toxicity & Addiction Scores", node_score),
    ("LLM Reasoning Agent", node_reason),
    ("Action Execution", node_action),
]


def _build_langgraph_app():
    graph = StateGraph(WatcherState)
    graph.add_node("ingest", node_ingest)
    graph.add_node("score", node_score)
    graph.add_node("reason", node_reason)
    graph.add_node("action", node_action)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "score")
    graph.add_edge("score", "reason")
    graph.add_edge("reason", "action")
    graph.add_edge("action", END)
    return graph.compile()


def run_agent_pipeline(activity_data: Dict[str, pd.DataFrame], rules: Optional[Dict[str, Any]] = None,
                        step_callback=None) -> WatcherState:
    """
    Run the 4-node agentic pipeline over a data bundle (as produced by
    data_simulator.load_preset). If `step_callback` is provided, it is
    called as step_callback(node_name, state) after each node executes,
    so the UI can render progressive node-by-node output.
    """
    rules = rules or DEFAULT_RULES
    initial_state: WatcherState = {
        "rules": rules,
        "screen_time_df": activity_data.get("screen_time", pd.DataFrame()),
        "app_usage_df": activity_data.get("app_usage", pd.DataFrame()),
        "messages_df": activity_data.get("messages", pd.DataFrame()),
        "activity_log_df": activity_data.get("activity_log", pd.DataFrame()),
    }

    if LANGGRAPH_AVAILABLE and step_callback is None:
        app = _build_langgraph_app()
        return app.invoke(initial_state)

    # Sequential fallback (also used when a step_callback is requested, so the
    # UI can show per-node progress regardless of whether langgraph is present).
    state = initial_state
    for name, fn in NODE_SEQUENCE:
        state = fn(state)
        if step_callback:
            step_callback(name, state)
    return state
