"""
app.py
------
"The Watcher" — AI-powered Child Digital Activity & Behavioral Safety
Monitoring Dashboard.

Run with:  streamlit run app.py
"""

import io
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import data_simulator as ds
import agent_engine as ae

# --------------------------------------------------------------------------
# Page config & theme
# --------------------------------------------------------------------------
st.set_page_config(page_title="The Watcher", page_icon="🛡️", layout="wide")

ACCENT = "#FF8C42"
DARK = "#2C3E50"
BG = "#FAFAF8"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {DARK}; }}
    section[data-testid="stSidebar"] * {{ color: #F5F5F5 !important; }}
    h1, h2, h3 {{ color: {DARK}; }}
    div[data-testid="stMetric"] {{
        background-color: white; border: 1px solid #eee; border-radius: 10px;
        padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; color: white; }}
    .badge-critical {{ background-color: #E74C3C; }}
    .badge-medium {{ background-color: {ACCENT}; }}
    .badge-low {{ background-color: #7F8C8D; }}
    .node-box {{
        background-color: white; border: 2px solid {ACCENT}; border-radius: 10px;
        padding: 14px; text-align: center; font-weight: 600; color: {DARK};
    }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_state():
    if "preset_data" not in st.session_state:
        st.session_state.preset_data = ds.load_preset("Normal Teen")
    if "rules" not in st.session_state:
        st.session_state.rules = dict(ae.DEFAULT_RULES)
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "node_log" not in st.session_state:
        st.session_state.node_log = []


init_state()


def run_pipeline():
    st.session_state.node_log = []

    def cb(name, state):
        st.session_state.node_log.append((name, dict(state)))

    result = ae.run_agent_pipeline(st.session_state.preset_data, st.session_state.rules, step_callback=cb)
    st.session_state.pipeline_result = result
    return result


def severity_badge(sev: str) -> str:
    cls = {"Critical": "badge-critical", "Medium": "badge-medium", "Low": "badge-low"}.get(sev, "badge-low")
    return f'<span class="badge {cls}">{sev}</span>'


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ The Watcher")
st.sidebar.caption("AI-powered child digital safety monitor — demo prototype using fully synthetic data.")
page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard & Analytics",
        "Activity Simulation & Ingestion",
        "Agentic Decision Rules & Configuration",
        "Intervention & Behavior Reports",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    "💡 Best practice: use tools like this transparently — most child-safety experts "
    "recommend discussing monitoring openly with your child rather than hiding it."
)

data = st.session_state.preset_data

# ==========================================================================
# PAGE 1 — Dashboard & Analytics
# ==========================================================================
if page == "Dashboard & Analytics":
    st.title("📊 Dashboard & Analytics")
    st.caption(f"Current dataset: **{data['preset_name']}** — switch scenarios on the Simulation page.")

    if st.button("🔄 Refresh Analysis", type="primary"):
        run_pipeline()

    result = st.session_state.pipeline_result or run_pipeline()

    screen_time_df = data["screen_time"]
    today_hours = screen_time_df["screen_time_hours"].iloc[-1]
    avg_hours = screen_time_df["screen_time_hours"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Screen Time (today)", f"{today_hours:.1f} h", delta=f"{today_hours-avg_hours:+.1f} h vs avg")
    col2.metric("Safety Score", f"{result['safety_score']:.0f}/100",
                delta=f"{result['safety_score']-70:+.0f} vs baseline", delta_color="normal")
    col3.metric("Triggered Alerts", len(result["alerts"]))
    col4.metric("Risk Category", result["risk_category"])

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Screen Time Over Time")
        fig = px.line(screen_time_df, x="date", y="screen_time_hours", markers=True,
                       color_discrete_sequence=[ACCENT])
        fig.add_hline(y=st.session_state.rules["max_screen_time_hours"], line_dash="dash",
                       line_color=DARK, annotation_text="Configured limit")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("App Category Usage")
        fig2 = px.pie(data["app_usage"], names="category", values="usage_pct",
                       color_discrete_sequence=px.colors.sequential.Oranges_r)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Risk Heatmap — Activity by Hour & Category")
    activity_log = data["activity_log"].copy()
    if not activity_log.empty:
        activity_log["hour"] = activity_log["timestamp"].apply(lambda t: t.hour)
        pivot = activity_log.pivot_table(index="category", columns="hour", values="duration_min",
                                          aggfunc="sum", fill_value=0)
        fig3 = px.imshow(pivot, aspect="auto", color_continuous_scale="Oranges",
                          labels=dict(x="Hour of Day", y="Category", color="Minutes"))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No activity log data available for this dataset.")

    st.subheader("🔔 Live Alert Feed")
    for alert in sorted(result["alerts"], key=lambda a: {"Critical": 0, "Medium": 1, "Low": 2}[a["severity"]]):
        st.markdown(
            f"{severity_badge(alert['severity'])} &nbsp; **{alert['category']}** "
            f"&nbsp;·&nbsp; {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>{alert['message']}",
            unsafe_allow_html=True,
        )
        st.markdown("")

# ==========================================================================
# PAGE 2 — Activity Simulation & Ingestion
# ==========================================================================
elif page == "Activity Simulation & Ingestion":
    st.title("🧪 Activity Simulation & Ingestion")

    st.subheader("Load a mock scenario")
    preset_name = st.selectbox("Preset dataset", list(ds.PRESETS.keys()),
                                index=list(ds.PRESETS.keys()).index(data["preset_name"]))
    if st.button("Load Preset", type="primary"):
        st.session_state.preset_data = ds.load_preset(preset_name)
        st.session_state.pipeline_result = None
        st.success(f"Loaded '{preset_name}' dataset.")
        st.rerun()

    st.divider()
    tabs = st.tabs(["Activity Log", "Messages", "Search Queries", "Manual Log Entry"])

    with tabs[0]:
        st.dataframe(data["activity_log"], use_container_width=True, height=320)

    with tabs[1]:
        st.dataframe(data["messages"], use_container_width=True, height=320)

    with tabs[2]:
        st.dataframe(data["searches"], use_container_width=True, height=320)

    with tabs[3]:
        st.markdown("Add a single activity event to test the pipeline against a specific scenario.")
        with st.form("manual_entry_form"):
            fc1, fc2, fc3 = st.columns(3)
            category = fc1.selectbox("App category", ds.APP_CATEGORIES)
            app_name = fc2.text_input("App name", value=ds.APP_NAMES[category][0])
            duration = fc3.number_input("Duration (minutes)", min_value=1, max_value=600, value=30)
            time_of_event = st.time_input("Time of event", value=dt.datetime.now().time())
            message_text = st.text_area("Optional message content (for toxicity scoring)", value="")
            submitted = st.form_submit_button("Add Entry")

        if submitted:
            ts = dt.datetime.combine(dt.date.today(), time_of_event)
            new_row = ds.new_manual_entry(app_name, category, duration, message_text, ts)
            new_df = pd.DataFrame([{
                "timestamp": new_row["timestamp"], "app": new_row["app"], "category": new_row["category"],
                "duration_min": new_row["duration_min"], "late_night": new_row["late_night"],
            }])
            st.session_state.preset_data["activity_log"] = pd.concat(
                [st.session_state.preset_data["activity_log"], new_df], ignore_index=True
            )
            if message_text.strip():
                is_concerning = any(p in message_text.lower() for p in ds.CONCERNING_PHRASES)
                new_msg = pd.DataFrame([{
                    "timestamp": ts, "app": app_name, "message_snippet": message_text,
                    "sentiment_score": -0.6 if is_concerning else 0.3, "concern_flag": is_concerning,
                }])
                st.session_state.preset_data["messages"] = pd.concat(
                    [st.session_state.preset_data["messages"], new_msg], ignore_index=True
                )
            st.session_state.pipeline_result = None
            st.success("Manual entry added. Re-run the pipeline on the Agentic Decision Engine page to see its effect.")

# ==========================================================================
# PAGE 3 — Agentic Decision Rules & Configuration
# ==========================================================================
elif page == "Agentic Decision Rules & Configuration":
    st.title("🤖 Agentic Decision Engine")

    backend_note = "LangGraph" if ae.LANGGRAPH_AVAILABLE else "sequential fallback (langgraph not installed)"
    llm_note = "OpenAI (langchain-openai detected)" if ae.LANGCHAIN_OPENAI_AVAILABLE else \
               ("Ollama (langchain-ollama detected)" if ae.LANGCHAIN_OLLAMA_AVAILABLE else
                "rule-based heuristic reasoning (no LLM backend detected / no API key set)")
    st.info(f"Workflow engine: **{backend_note}**  ·  Reasoning backend available: **{llm_note}**")

    st.subheader("Workflow")
    n1, arrow1, n2, arrow2, n3, arrow3, n4 = st.columns([3, 1, 3, 1, 3, 1, 3])
    with n1:
        st.markdown('<div class="node-box">1️⃣ Ingest Activity Stream</div>', unsafe_allow_html=True)
    with arrow1:
        st.markdown("<h3 style='text-align:center'>→</h3>", unsafe_allow_html=True)
    with n2:
        st.markdown('<div class="node-box">2️⃣ Toxicity & Addiction Scoring</div>', unsafe_allow_html=True)
    with arrow2:
        st.markdown("<h3 style='text-align:center'>→</h3>", unsafe_allow_html=True)
    with n3:
        st.markdown('<div class="node-box">3️⃣ LLM Reasoning Agent</div>', unsafe_allow_html=True)
    with arrow3:
        st.markdown("<h3 style='text-align:center'>→</h3>", unsafe_allow_html=True)
    with n4:
        st.markdown('<div class="node-box">4️⃣ Action Execution</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Rule Configuration")
    rules = st.session_state.rules
    rc1, rc2 = st.columns(2)
    with rc1:
        rules["max_screen_time_hours"] = st.slider("Max daily screen time (hours)", 1.0, 12.0,
                                                     float(rules["max_screen_time_hours"]), 0.5)
        rules["toxicity_alert_threshold"] = st.slider("Toxicity alert threshold", 0, 100,
                                                        int(rules["toxicity_alert_threshold"]))
        rules["addiction_alert_threshold"] = st.slider("Addiction alert threshold", 0, 100,
                                                         int(rules["addiction_alert_threshold"]))
    with rc2:
        rules["bedtime_start"] = st.time_input("Bedtime starts", value=rules["bedtime_start"])
        rules["bedtime_end"] = st.time_input("Bedtime ends", value=rules["bedtime_end"])
        rules["blocked_categories_at_night"] = st.multiselect(
            "Blocked categories during bedtime", ds.APP_CATEGORIES,
            default=rules["blocked_categories_at_night"],
        )

    if st.button("💾 Save Rules"):
        st.session_state.rules = rules
        st.success("Rules saved.")

    st.divider()
    if st.button("▶️ Run Agent Pipeline", type="primary"):
        run_pipeline()

    if st.session_state.node_log:
        st.subheader("Node-by-node execution trace")
        for name, state in st.session_state.node_log:
            with st.expander(name, expanded=(name == "Action Execution")):
                if name == "Ingest Activity Stream":
                    st.write(f"Activity log rows: {len(state['activity_log_df'])}  ·  "
                             f"Messages: {len(state['messages_df'])}  ·  "
                             f"Screen-time days: {len(state['screen_time_df'])}")
                elif name == "Calculate Toxicity & Addiction Scores":
                    st.write(f"Toxicity score: **{state['toxicity_score']}/100**")
                    st.write(f"Addiction score: **{state['addiction_score']}/100**")
                    st.write(f"Safety score: **{state['safety_score']}/100**")
                    st.write(f"Risk category: **{state['risk_category']}**")
                elif name == "LLM Reasoning Agent":
                    st.write(f"Backend used: `{state['llm_backend_used']}`")
                    st.write(state["reasoning_text"])
                elif name == "Action Execution":
                    for alert in state["alerts"]:
                        st.markdown(f"{severity_badge(alert['severity'])} **{alert['category']}** — {alert['message']}",
                                    unsafe_allow_html=True)

# ==========================================================================
# PAGE 4 — Intervention & Behavior Reports
# ==========================================================================
elif page == "Intervention & Behavior Reports":
    st.title("📝 Intervention & Behavior Reports")

    result = st.session_state.pipeline_result
    if result is None:
        st.warning("No analysis has been run yet. Go to **Agentic Decision Rules & Configuration** and click "
                   "'Run Agent Pipeline' first.")
        st.stop()

    st.subheader("Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Safety Score", f"{result['safety_score']:.0f}/100")
    s2.metric("Toxicity Score", f"{result['toxicity_score']:.0f}/100")
    s3.metric("Addiction Score", f"{result['addiction_score']:.0f}/100")
    st.markdown(f"**Risk category:** {result['risk_category']}")

    st.subheader("AI Reasoning")
    st.write(result["reasoning_text"])
    st.caption(f"Reasoning backend: {result['llm_backend_used']}")

    st.subheader("Actionable Insights")
    for alert in result["alerts"]:
        st.markdown(f"{severity_badge(alert['severity'])} {alert['message']}", unsafe_allow_html=True)

    st.divider()
    st.subheader("Export Report")

    # --- CSV export ---
    alerts_df = pd.DataFrame(result["alerts"])
    csv_buf = io.StringIO()
    alerts_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download Alerts (CSV)", data=csv_buf.getvalue(),
                        file_name="watcher_alerts.csv", mime="text/csv")

    # --- PDF export ---
    def build_pdf_bytes(result: dict) -> bytes:
        try:
            from fpdf import FPDF
        except ImportError:
            return b""

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "The Watcher - Behavior & Psychological Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {result['report']['generated_at'].strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Scores", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 7,
            f"Safety Score: {result['safety_score']}/100\n"
            f"Toxicity Score: {result['toxicity_score']}/100\n"
            f"Addiction Score: {result['addiction_score']}/100\n"
            f"Risk Category: {result['risk_category']}"
        )
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Reasoning", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 7, result["reasoning_text"])
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Recommended Actions", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for alert in result["alerts"]:
            pdf.multi_cell(0, 7, f"[{alert['severity']}] {alert['category']}: {alert['message']}")

        return bytes(pdf.output(dest="S"))

    pdf_bytes = build_pdf_bytes(result)
    if pdf_bytes:
        st.download_button("⬇️ Download Full Report (PDF)", data=pdf_bytes,
                            file_name="watcher_report.pdf", mime="application/pdf")
    else:
        st.caption("Install `fpdf2` (see requirements.txt) to enable PDF export.")
