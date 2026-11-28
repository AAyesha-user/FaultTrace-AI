from embedded_api import ensure_api_running
from api_page import render_api_page
import time
from backend.ai.explainer import generate_report
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.simulation.engine import WarehouseSimulation
from backend.diagnosis.collaborative import (
    diagnose_collaboratively,
    explain_consensus,
)
from backend.recovery.engine import simulate_recovery


# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

EXPERIMENTS = DATA / "experiments"
MULTI = DATA / "multi_fault"
STRESS = DATA / "stress_tests"

ROBOTS = ["R1", "R2", "R3", "R4", "R5"]

st.set_page_config(
    page_title="FaultTrace AI | Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_STATUS = ensure_api_running()


# ==========================================================
# 2. DESIGN SYSTEM  (Midnight · Indigo · Violet · Cyan)
# ==========================================================

st.markdown("""
<style>

:root {
    --background: #060612;
    --panel: #0E0F26;
    --panel-light: #161838;
    --border: rgba(139,92,246,0.25);
    --violet: #8B5CF6;
    --indigo: #6366F1;
    --cyan: #22D3EE;
    --text: #EEF0FF;
    --muted: #9CA3D4;
}

.stApp {
    background:
        radial-gradient(circle at 90% 0%, rgba(139,92,246,0.18), transparent 32%),
        radial-gradient(circle at 5% 90%, rgba(34,211,238,0.07), transparent 30%),
        linear-gradient(135deg, #060612, #0A0B1F 60%, #070713);
    color: var(--text);
}

.block-container {
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: #F4F3FF !important;
}

p, label, li {
    color: #C4C9EA;
}

[data-testid="stCaptionContainer"] {
    color: #9CA3D4 !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E0F26, #0A0B1F 65%, #070713);
    border-right: 1px solid rgba(139,92,246,0.25);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 12px;
    border-radius: 10px;
    transition: 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(139,92,246,0.16);
}

/* BRAND */

.brand-mark {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 8px;
}

.brand-symbol {
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    color: white;
    font-size: 24px;
    font-weight: 900;
    background: linear-gradient(135deg, #6366F1, #8B5CF6 55%, #22D3EE);
    box-shadow: 0 0 26px rgba(139,92,246,0.45);
}

.brand-name {
    font-size: 21px;
    font-weight: 850;
    color: #F4F3FF;
}

.brand-sub {
    font-size: 9px;
    color: #9CA3D4;
    letter-spacing: 2px;
    margin-top: 3px;
}

/* HEADERS */

.page-title {
    background: linear-gradient(90deg, #FFFFFF, #C4B5FD 60%, #67E8F9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: clamp(31px, 4vw, 43px);
    font-weight: 850;
    letter-spacing: -1.5px;
    line-height: 1.15;
    margin-bottom: 12px;
}

.page-description {
    color: #B4BBE0;
    font-size: 15px;
    line-height: 1.8;
    max-width: 850px;
}

.hero-panel {
    padding: 34px 36px;
    border-radius: 20px;
    background:
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.30), transparent 38%),
        radial-gradient(circle at 10% 100%, rgba(34,211,238,0.08), transparent 35%),
        linear-gradient(120deg, #1A1B40, #0E0F26 70%, #0A0B1F);
    border: 1px solid rgba(139,92,246,0.32);
    box-shadow: 0 15px 45px rgba(76,29,149,0.22);
    margin-bottom: 26px;
}

.hero-chip {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 20px;
    color: #C4B5FD;
    background: rgba(139,92,246,0.16);
    border: 1px solid rgba(167,139,250,0.30);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.6px;
    margin-bottom: 15px;
}

/* SECTION HEADINGS */

.section-heading {
    font-size: 21px;
    font-weight: 750;
    color: #F4F3FF;
    margin-bottom: 5px;
}

.section-description {
    font-size: 13px;
    color: #9CA3D4;
    margin-bottom: 18px;
}

.section-label {
    color: #A78BFA;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-top: 10px;
    margin-bottom: 12px;
}

/* KPI CARDS */

.kpi-card {
    background: linear-gradient(145deg, #181A3E, #0E0F26);
    border: 1px solid rgba(139,92,246,0.24);
    border-radius: 16px;
    padding: 20px;
    min-height: 133px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    transition: 0.2s ease;
}

.kpi-card:hover {
    border-color: rgba(167,139,250,0.65);
    box-shadow: 0 8px 30px rgba(139,92,246,0.22);
    transform: translateY(-2px);
}

.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.kpi-label {
    color: #A5ACD8;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}

.kpi-icon {
    color: #22D3EE;
    font-size: 20px;
}

.kpi-value {
    font-size: 31px;
    font-weight: 850;
    color: #F4F3FF;
    margin-top: 13px;
}

.kpi-note {
    color: #8D95C7;
    font-size: 11px;
    margin-top: 5px;
}

/* FEATURE CARDS */

.feature-card {
    min-height: 145px;
    padding: 21px;
    border-radius: 15px;
    background: linear-gradient(145deg, #181A3E, #0E0F26);
    border: 1px solid rgba(139,92,246,0.20);
    transition: 0.2s ease;
}

.feature-card:hover {
    border-color: rgba(167,139,250,0.60);
    box-shadow: 0 8px 28px rgba(139,92,246,0.18);
    transform: translateY(-2px);
}

.feature-icon {
    font-size: 25px;
    color: #A78BFA;
    margin-bottom: 12px;
}

.feature-title {
    color: #F4F3FF;
    font-size: 16px;
    font-weight: 750;
    margin-bottom: 8px;
}

.feature-description {
    color: #A5ACD8;
    font-size: 12px;
    line-height: 1.7;
}

/* STATUS */

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border: 1px solid rgba(34,211,238,0.30);
    background: rgba(34,211,238,0.08);
    color: #67E8F9;
    padding: 7px 11px;
    border-radius: 30px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22D3EE;
    box-shadow: 0 0 8px #22D3EE;
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid rgba(167,139,250,0.55);
    background: linear-gradient(110deg, #6D28D9, #4F46E5);
    color: white !important;
    font-weight: 750;
    min-height: 43px;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #C4B5FD;
    background: linear-gradient(110deg, #7C3AED, #06B6D4);
    box-shadow: 0 5px 22px rgba(139,92,246,0.40);
}

/* INPUTS */

[data-baseweb="select"] > div {
    background: #161838 !important;
    border-color: rgba(139,92,246,0.32) !important;
    color: #F4F3FF !important;
}

.stTextInput input,
.stNumberInput input {
    background: #161838 !important;
    color: #F4F3FF !important;
    border-color: rgba(139,92,246,0.32) !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #0E0F26 !important;
}

[role="option"] {
    background: #0E0F26 !important;
    color: #F4F3FF !important;
}

[role="option"]:hover {
    background: #231F52 !important;
}

/* TABLES AND EXPANDERS */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid rgba(139,92,246,0.24);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    background: #0E0F26;
    border: 1px solid rgba(139,92,246,0.24);
    border-radius: 12px;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #161838;
    border: 1px solid rgba(139,92,246,0.24);
    border-radius: 12px;
    padding: 13px;
}

[data-testid="stMetricValue"] {
    color: #F4F3FF !important;
}

[data-testid="stMetricLabel"] {
    color: #A5ACD8 !important;
}

/* DIVIDERS AND PROGRESS */

hr {
    border-color: rgba(139,92,246,0.20);
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #8B5CF6, #22D3EE);
}

/* SCROLLBAR */

::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}

::-webkit-scrollbar-track {
    background: #0A0B1F;
}

::-webkit-scrollbar-thumb {
    background: #3B2F7A;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #8B5CF6;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. REUSABLE UI COMPONENTS
# ==========================================================

def page_header(eyebrow, title, description):
    st.markdown(f"""
    <div class="hero-panel">
        <div class="hero-chip">{eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def section_heading(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )

    if description:
        st.markdown(
            f'<div class="section-description">{description}</div>',
            unsafe_allow_html=True
        )


def kpi(label, value, note="", icon="◈"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-top">
            <div class="kpi-label">{label}</div>
            <div class="kpi-icon">{icon}</div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def feature(icon, title, description):
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not load {path.name}: {error}")
        return pd.DataFrame()


def normalize_success(df):
    df = df.copy()

    if "success" in df.columns:
        if df["success"].dtype != bool:
            df["success"] = (
                df["success"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                })
            )

    return df


def style_chart(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            color="#C4C9EA",
            size=12
        ),
        height=height,
        margin=dict(l=15, r=25, t=55, b=25),
        colorway=[
            "#8B5CF6",
            "#22D3EE",
            "#6366F1",
            "#A78BFA",
            "#67E8F9"
        ],
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#C4C9EA")
        ),
        xaxis=dict(
            gridcolor="rgba(165,172,216,0.10)",
            zerolinecolor="rgba(165,172,216,0.14)"
        ),
        yaxis=dict(
            gridcolor="rgba(165,172,216,0.10)",
            zerolinecolor="rgba(165,172,216,0.14)"
        )
    )

    return fig


def fleet_performance(df):
    df = normalize_success(df)

    if df.empty:
        return pd.DataFrame()

    if "robot_id" not in df.columns or "success" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["robot_id", "success"])

    if df.empty:
        return pd.DataFrame()

    result = df.groupby("robot_id").agg(
        total_tasks=("success", "count"),
        successful_tasks=("success", "sum")
    ).reset_index()

    result["failed_tasks"] = (
        result["total_tasks"] - result["successful_tasks"]
    )

    result["success_rate"] = (
        result["successful_tasks"] / result["total_tasks"]
    )

    result["failure_rate"] = (
        result["failed_tasks"] / result["total_tasks"]
    )

    return result


def show_table(df):
    if df is None or df.empty:
        st.info("No records available for this view.")
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


def safe_mean(df, column):
    if df.empty or column not in df.columns:
        return None

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).mean()


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.1%}"


def show_recovery_summary(summary):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Failed Tasks",
        summary.get("failed_tasks_identified", 0)
    )

    c2.metric(
        "Tasks Reassigned",
        summary.get("tasks_reassigned", 0)
    )

    c3.metric(
        "Successful Recoveries",
        summary.get("successful_recoveries", 0)
    )

    rate = float(summary.get("recovery_rate", 0))

    c4.metric(
        "Recovery Rate",
        f"{rate:.1f}%"
    )

    st.progress(max(0.0, min(rate / 100, 1.0)))


def download_csv(df, filename, label):
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
        use_container_width=True
    )


# ==========================================================
# 4. LOAD RESEARCH DATA
# ==========================================================

baseline_history = load_csv(
    EXPERIMENTS / "execution_history.csv"
)

stress_history = load_csv(
    STRESS / "execution_history.csv"
)

multi_history = load_csv(
    MULTI / "execution_history.csv"
)

baseline_evaluation = load_csv(
    EXPERIMENTS / "evaluation_per_run.csv"
)

baseline_rankings = load_csv(
    EXPERIMENTS / "diagnostic_rankings.csv"
)

stress_rankings = load_csv(
    STRESS / "diagnostic_rankings.csv"
)

multi_rankings = load_csv(
    MULTI / "diagnostic_rankings.csv"
)

stress_summary = load_csv(
    STRESS / "scenario_summary.csv"
)

multi_summary = load_csv(
    MULTI / "scenario_summary.csv"
)


# ==========================================================
# 5. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown("""
    <div class="brand-mark">
        <div class="brand-symbol">◈</div>
        <div>
            <div class="brand-name">FaultTrace AI</div>
            <div class="brand-sub">AUTONOMOUS DIAGNOSTICS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"API server: {API_STATUS}")

    st.markdown("---")

    st.markdown(
        '<div class="section-label">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Main navigation",
        [
            "Fleet Overview",
            "Live Simulator",
            "Diagnosis Explorer",
            "Recovery Center",
            "Experiment Analytics",
            "Execution History",
            "API Integration"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">SYSTEM STATUS</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="status-pill">
        <span class="status-dot"></span>
        DIAGNOSTIC ENGINE READY
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("FAULTTRACE AI · VERSION 3.0")
    st.caption("Research-driven multi-agent fault diagnosis")

    st.markdown("---")

    st.caption(
        "Built for intelligent fault detection and autonomous recovery."
    )


# ==========================================================
# 6. FLEET OVERVIEW
# ==========================================================

if page == "Fleet Overview":

    page_header(
        "FAULTTRACE INTELLIGENCE PLATFORM",
        "Review fleet performance and baseline results",
        "An intelligent diagnostic environment for monitoring "
        "robotic agents, investigating failures and evaluating "
        "autonomous recovery."
    )

    total_tasks = len(baseline_history)
    total_runs = len(baseline_evaluation)
    total_stress = len(stress_summary)
    total_multi = len(multi_summary)

    section_heading(
        "Platform at a glance",
        "Your experimental environment and diagnostic research activity."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("ROBOT FLEET", "05", "Simulated autonomous agents", "⬡")

    with c2:
        kpi("BASELINE TASKS", f"{total_tasks:,}", "Recorded execution events", "◈")

    with c3:
        kpi("EXPERIMENT RUNS", f"{total_runs:,}", "Baseline evaluation runs", "◎")

    with c4:
        kpi(
            "FAULT SCENARIOS",
            f"{total_stress + total_multi:,}",
            "Stress and multi-fault scenarios",
            "⌁"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section_heading(
        "Intelligence modules",
        "Explore the capabilities of the FaultTrace diagnostic ecosystem."
    )

    a, b, c = st.columns(3)

    with a:
        feature(
            "◈",
            "Fleet Monitoring",
            "Observe robotic task execution and compare the performance "
            "of individual agents."
        )

    with b:
        feature(
            "⬡",
            "Fault Diagnosis",
            "Rank suspicious agents using diagnostic evidence "
            "and collaborative analysis."
        )

    with c:
        feature(
            "⌁",
            "Autonomous Recovery",
            "Investigate failed tasks and simulate reassignment "
            "to available agents."
        )

    a, b, c = st.columns(3)

    with a:
        feature(
            "⚡",
            "Fault Injection",
            "Create controlled intermittent and permanent fault scenarios."
        )

    with b:
        feature(
            "◎",
            "Multi-Fault Analysis",
            "Evaluate diagnostic performance when multiple agents are faulty."
        )

    with c:
        feature(
            "▤",
            "Research Analytics",
            "Review experimental metrics and diagnostic ranking performance."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section_heading(
        "Fleet performance",
        "Historical baseline execution performance."
    )

    performance = fleet_performance(baseline_history)

    if not performance.empty:

        left, right = st.columns([1.35, 1])

        with left:

            fig = px.bar(
                performance,
                x="robot_id",
                y=["successful_tasks", "failed_tasks"],
                barmode="group",
                title="Robot Task Outcomes",
                labels={
                    "robot_id": "Robot",
                    "value": "Tasks",
                    "variable": "Outcome"
                }
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        with right:

            fig = px.bar(
                performance,
                x="robot_id",
                y="success_rate",
                title="Robot Success Rate",
                text="success_rate"
            )

            fig.update_traces(
                texttemplate="%{text:.1%}",
                textposition="outside"
            )

            fig.update_yaxes(range=[0, 1.15])

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        show_table(performance)

    else:
        st.info("Baseline fleet history has not been generated yet.")

    if not baseline_evaluation.empty:

        st.markdown("<br>", unsafe_allow_html=True)

        section_heading(
            "Baseline diagnostic quality",
            "Aggregate evaluation of the baseline diagnostic engine."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Top-1 Accuracy",
            format_percent(safe_mean(baseline_evaluation, "top1"))
        )

        c2.metric(
            "Top-3 Accuracy",
            format_percent(safe_mean(baseline_evaluation, "top3"))
        )

        mrr = safe_mean(baseline_evaluation, "reciprocal_rank")

        c3.metric(
            "Mean Reciprocal Rank",
            f"{mrr:.4f}" if mrr is not None and not pd.isna(mrr) else "N/A"
        )


# ==========================================================
# 7. LIVE SIMULATOR
# ==========================================================

elif page == "Live Simulator":

    page_header(
        "LIVE SIMULATION CONTROL",
        "Simulate robot faults and diagnose them live",
        "Configure a robotic fleet experiment and evaluate "
        "FaultTrace's collaborative diagnostic and recovery workflow."
    )

    section_heading(
        "Experiment configuration",
        "Choose which robots become faulty and configure their behaviour."
    )

    left, right = st.columns([1.05, 0.95])

    with left:

        with st.container(border=True):

            st.markdown("#### 01 · Robot selection")

            faulty_robots = st.multiselect(
                "Select faulty robots",
                ROBOTS,
                default=["R3"],
                help="Choose one or multiple robots."
            )

            st.markdown("#### 02 · Fault characteristics")

            fault_mode = st.selectbox(
                "Fault behaviour",
                ["Intermittent", "Permanent"]
            )

            fault_probability = st.slider(
                "Activation probability",
                0.05,
                1.0,
                0.30,
                0.05
            )

            fault_severity = st.slider(
                "Fault severity",
                0.05,
                0.95,
                0.70,
                0.05
            )

    with right:

        with st.container(border=True):

            st.markdown("#### 03 · Simulation settings")

            num_tasks = st.select_slider(
                "Number of tasks",
                options=[50, 100, 150, 200, 300, 400, 500],
                value=100
            )

            seed = st.number_input(
                "Random seed",
                min_value=1,
                value=42,
                step=1
            )

            st.markdown("---")

            st.markdown("#### Experiment summary")

            st.write("**Fleet size:** 5 robots")
            st.write(f"**Faulty agents:** {len(faulty_robots)}")
            st.write(f"**Fault mode:** {fault_mode}")
            st.write(f"**Task volume:** {num_tasks}")
            st.write(f"**Severity:** {fault_severity:.0%}")

            st.caption(
                "Permanent mode uses an activation probability of 100%. "
                "Intermittent mode uses your selected probability."
            )

    effective_probability = (
        1.0 if fault_mode == "Permanent"
        else fault_probability
    )

    st.markdown("<br>", unsafe_allow_html=True)

    run_col, info_col = st.columns([1, 2])

    with run_col:

        run_simulation = st.button(
            "⚡ EXECUTE SIMULATION",
            use_container_width=True
        )

    with info_col:

        st.caption(
            "The experiment generates execution records, runs "
            "collaborative diagnosis and stores results in the current session."
        )

    if run_simulation:

        if not faulty_robots:

            st.error("Select at least one faulty robot.")

        else:

            try:

                with st.spinner(
                    "Executing robotic tasks and analyzing failures..."
                ):

                    simulation = WarehouseSimulation(
                        num_robots=5,
                        seed=int(seed),
                        faulty_robot_ids=faulty_robots,
                        fault_probability=effective_probability,
                        fault_mode=fault_mode,
                        fault_severity=fault_severity
                    )

                    simulation.generate_tasks(num_tasks=num_tasks)

                    records = simulation.run()

                    df = pd.DataFrame(
                        [record.to_dict() for record in records]
                    )

                    df = normalize_success(df)

                    if df.empty:
                        raise ValueError(
                            "The simulation generated no records."
                        )

                    if "success" not in df.columns:
                        raise ValueError(
                            "Execution records do not contain a success column."
                        )

                    diagnosis = diagnose_collaboratively(df)

                    explanations = explain_consensus(diagnosis)

                    st.session_state["live_history"] = df
                    st.session_state["live_diagnosis"] = diagnosis
                    st.session_state["live_explanations"] = explanations
                    st.session_state["live_faults"] = list(faulty_robots)
                    st.session_state["live_mode"] = fault_mode
                    st.session_state["live_probability"] = effective_probability
                    st.session_state["live_severity"] = fault_severity
                    st.session_state["live_recovery"] = None
                    st.session_state.pop("live_recovery_summary", None)
                    st.session_state.pop("ai_report", None)

                st.success("Simulation completed successfully.")

            except Exception as error:

                st.error(f"Simulation error: {error}")

    if "live_history" in st.session_state:

        df = st.session_state["live_history"]
        diagnosis = st.session_state["live_diagnosis"]
        actual_faults = st.session_state["live_faults"]

        st.markdown("---")

        section_heading(
            "Simulation results",
            "Execution summary generated by the robotic environment."
        )

        success_count = int(df["success"].sum())
        failed_count = len(df) - success_count
        success_rate = success_count / len(df) if len(df) else 0

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi(
                "TASKS EXECUTED",
                f"{len(df):,}",
                "Total simulation workload",
                "▤"
            )

        with c2:
            kpi(
                "SUCCESSFUL",
                f"{success_count:,}",
                "Completed operations",
                "✓"
            )

        with c3:
            kpi(
                "FAILED",
                f"{failed_count:,}",
                "Unsuccessful operations",
                "!"
            )

        with c4:
            kpi(
                "SUCCESS RATE",
                f"{success_rate:.1%}",
                "Overall execution performance",
                "◎"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([1.2, 0.8])

        with left:

            performance = fleet_performance(df)

            if not performance.empty:

                fig = px.bar(
                    performance,
                    x="robot_id",
                    y=["successful_tasks", "failed_tasks"],
                    barmode="group",
                    title="Robot-level task outcomes"
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

        with right:

            st.markdown("#### Experiment configuration")

            st.write("**Faulty robots:** " + ", ".join(actual_faults))
            st.write(f"**Fault mode:** {st.session_state['live_mode']}")
            st.write(
                f"**Activation:** "
                f"{st.session_state['live_probability']:.0%}"
            )
            st.write(
                f"**Severity:** "
                f"{st.session_state['live_severity']:.0%}"
            )

        st.markdown("---")

        section_heading(
            "Collaborative diagnosis",
            "Diagnostic ranking produced by the collaborative engine."
        )

        if diagnosis is not None and not diagnosis.empty:

            left, right = st.columns([1.15, 0.85])

            score_col = (
                "consensus_score"
                if "consensus_score" in diagnosis.columns
                else None
            )

            with left:

                if score_col and "robot_id" in diagnosis.columns:

                    chart_data = diagnosis.copy()

                    chart_data[score_col] = pd.to_numeric(
                        chart_data[score_col],
                        errors="coerce"
                    )

                    chart_data = chart_data.dropna(subset=[score_col])

                    if not chart_data.empty:

                        chart_data = chart_data.sort_values(
                            score_col,
                            ascending=True
                        )

                        fig = px.bar(
                            chart_data,
                            x=score_col,
                            y="robot_id",
                            orientation="h",
                            title="Collaborative suspicion ranking",
                            text=score_col
                        )

                        fig.update_traces(
                            texttemplate="%{text:.4f}",
                            textposition="outside"
                        )

                        st.plotly_chart(
                            style_chart(fig),
                            use_container_width=True
                        )

                    else:
                        st.info(
                            "No valid consensus scores are available."
                        )

                else:
                    st.info(
                        "No consensus score column was returned."
                    )

            with right:

                st.markdown("#### Diagnostic ranking")
                show_table(diagnosis)

            if "robot_id" in diagnosis.columns:

                predicted = diagnosis.iloc[0]["robot_id"]

                if predicted in actual_faults:

                    st.success(
                        f"Top-ranked robot {predicted} matches an injected fault."
                    )

                else:

                    st.warning(
                        f"Top-ranked robot: {predicted}. "
                        f"Injected faults: {actual_faults}"
                    )

        else:

            st.warning(
                "The diagnostic engine returned no ranking records."
            )

        # ======================================================
        # AI DIAGNOSTIC REPORT
        # ======================================================

        st.markdown("---")

        section_heading(
            "AI diagnostic report",
            "Plain-English explanation of the diagnosis."
        )

        if st.button(
            "🧠 GENERATE AI REPORT",
            use_container_width=True
        ):

            try:
                key = st.secrets["OPENROUTER_API_KEY"]
            except Exception:
                key = None

            with st.spinner("Writing report..."):

                try:

                    st.session_state["ai_report"] = generate_report(
                        diagnosis,
                        injected_faults=actual_faults,
                        config={
                            "mode": st.session_state["live_mode"],
                            "severity": st.session_state["live_severity"],
                        },
                        api_key=key,
                    )

                except Exception as error:

                    st.error(f"AI report error: {error}")

        report = st.session_state.get("ai_report")

        if report:

            st.info(report["text"])

            if report["source"] == "fallback":
                st.caption(
                    "Offline summary shown (AI service unavailable)."
                )

        # ======================================================
        # DIAGNOSTIC AGENT EXPLANATIONS
        # ======================================================

        with st.expander(
            "Inspect diagnostic agent explanations"
        ):
            show_table(st.session_state["live_explanations"])

        with st.expander(
            "Inspect complete execution history"
        ):
            show_table(df)

        st.markdown("---")

        st.markdown("---")

        section_heading(
            "Live execution replay",
            "Replays the simulation task by task, like a live feed."
        )

        speed = st.select_slider(
            "Replay speed",
            options=["Slow", "Normal", "Fast"],
            value="Normal"
        )

        if st.button("▶ START LIVE REPLAY", use_container_width=True):

            delay = {"Slow": 0.15, "Normal": 0.05, "Fast": 0.01}[speed]
            bar = st.progress(0.0)
            feed = st.empty()
            tally = {r: {"tasks": 0, "failed": 0} for r in sorted(df["robot_id"].unique())}

            for i, rec in enumerate(df.to_dict("records"), start=1):
                tally[rec["robot_id"]]["tasks"] += 1
                if not bool(rec["success"]):
                    tally[rec["robot_id"]]["failed"] += 1

                if i % 5 == 0 or i == len(df):
                    live = pd.DataFrame(tally).T.rename_axis("robot").reset_index()
                    live["failure_rate"] = (live["failed"] / live["tasks"]).fillna(0).round(2)
                    feed.dataframe(live, hide_index=True, use_container_width=True)
                    bar.progress(i / len(df))

                time.sleep(delay)
                
        # ======================================================
        # AUTONOMOUS RECOVERY
        # ======================================================

        section_heading(
            "Autonomous recovery",
            "Isolate selected robots and simulate failed-task reassignment."
        )

        available = (
            diagnosis["robot_id"].tolist()
            if not diagnosis.empty and "robot_id" in diagnosis.columns
            else []
        )

        selected_suspects = st.multiselect(
            "Select robots to isolate",
            available,
            default=available[:1],
            key="live_recovery_selection"
        )

        if st.button(
            "⚡ RUN RECOVERY SIMULATION",
            use_container_width=True
        ):

            if not selected_suspects:

                st.error("Select at least one robot.")

            else:

                try:

                    with st.spinner("Simulating task reassignment..."):

                        recovered, summary = simulate_recovery(
                            df,
                            selected_suspects
                        )

                        st.session_state["live_recovery"] = recovered
                        st.session_state["live_recovery_summary"] = summary
                        st.session_state["live_isolated"] = selected_suspects

                    st.success("Recovery simulation completed.")

                except Exception as error:

                    st.error(f"Recovery error: {error}")

        if st.session_state.get("live_recovery") is not None:

            st.markdown("#### Recovery performance")

            show_recovery_summary(
                st.session_state["live_recovery_summary"]
            )

            st.markdown("#### Reassigned task records")

            recovered = st.session_state["live_recovery"]

            show_table(recovered)

            download_csv(
                recovered,
                "faulttrace_recovery_report.csv",
                "Download recovery report"
            )


# ==========================================================
# 8. DIAGNOSIS EXPLORER
# ==========================================================

elif page == "Diagnosis Explorer":

    page_header(
        "DIAGNOSTIC INTELLIGENCE",
        "Identify which robots are most likely faulty",
        "Explore diagnostic scores, suspiciousness rankings "
        "and collaborative agent evaluations across experiments."
    )

    datasets = {}

    if not baseline_rankings.empty:
        datasets["Baseline diagnosis"] = baseline_rankings

    if not stress_rankings.empty:
        datasets["Stress-test diagnosis"] = stress_rankings

    if not multi_rankings.empty:
        datasets["Multi-fault diagnosis"] = multi_rankings

    if "live_diagnosis" in st.session_state:
        datasets["Live collaborative diagnosis"] = (
            st.session_state["live_diagnosis"]
        )

    if not datasets:

        st.warning("No diagnostic ranking data found.")

    else:

        left, right = st.columns(2)

        with left:

            dataset_name = st.selectbox(
                "Diagnostic dataset",
                list(datasets.keys())
            )

        df = datasets[dataset_name].copy()

        with right:

            score_options = [
                col for col in [
                    "consensus_score",
                    "suspiciousness",
                    "diagnostic_score"
                ]
                if col in df.columns
            ]

            if score_options:

                score_col = st.selectbox(
                    "Ranking metric",
                    score_options
                )

            else:

                score_col = "No available score"

        if "scenario" in df.columns:

            scenarios = sorted(
                df["scenario"].dropna().unique().tolist()
            )

            if scenarios:

                scenario = st.selectbox(
                    "Experiment scenario",
                    ["All"] + scenarios
                )

                if scenario != "All":
                    df = df[df["scenario"] == scenario]

        if "run_id" in df.columns:

            run_ids = sorted(
                df["run_id"].dropna().unique().tolist()
            )

            if run_ids:

                selected_run = st.selectbox(
                    "Experiment run",
                    ["All"] + run_ids
                )

                if selected_run != "All":
                    df = df[df["run_id"] == selected_run]

        st.markdown("---")

        if (
            score_col != "No available score"
            and score_col in df.columns
            and "robot_id" in df.columns
        ):

            df[score_col] = pd.to_numeric(
                df[score_col],
                errors="coerce"
            )

            valid_df = df.dropna(
                subset=["robot_id", score_col]
            ).copy()

            if not valid_df.empty:

                section_heading(
                    "Suspiciousness ranking",
                    "Higher scores indicate greater diagnostic suspicion. "
                    "Repeated records for a robot are combined using their mean score."
                )

                ranking_data = (
                    valid_df
                    .groupby("robot_id", as_index=False)
                    .agg(**{score_col: (score_col, "mean")})
                )

                ranking_data = ranking_data.sort_values(
                    score_col,
                    ascending=False
                ).reset_index(drop=True)

                chart_data = ranking_data.sort_values(
                    score_col,
                    ascending=True
                )

                fig = px.bar(
                    chart_data,
                    x=score_col,
                    y="robot_id",
                    orientation="h",
                    color=score_col,
                    color_continuous_scale=[
                        "#4F46E5",
                        "#8B5CF6",
                        "#A78BFA",
                        "#22D3EE"
                    ],
                    title="Robot Diagnostic Ranking",
                    text=score_col,
                    labels={
                        "robot_id": "Robot",
                        score_col: "Mean Suspiciousness Score"
                    }
                )

                fig.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_layout(
                    coloraxis_showscale=False,
                    yaxis=dict(
                        title="Robot",
                        categoryorder="total ascending"
                    ),
                    xaxis=dict(
                        title="Mean Suspiciousness Score",
                        rangemode="tozero"
                    ),
                    margin=dict(
                        l=25,
                        r=55,
                        t=65,
                        b=35
                    )
                )

                st.plotly_chart(
                    style_chart(fig, height=390),
                    use_container_width=True
                )

                top_robot = ranking_data.iloc[0]["robot_id"]
                top_score = ranking_data.iloc[0][score_col]

                c1, c2, c3 = st.columns(3)

                c1.metric("Top-ranked robot", str(top_robot))
                c2.metric("Highest diagnostic score", f"{top_score:.4f}")
                c3.metric("Robots evaluated", len(ranking_data))

                st.markdown("#### Diagnostic ranking details")

                display_ranking = ranking_data.copy()

                display_ranking.insert(
                    0,
                    "rank",
                    range(1, len(display_ranking) + 1)
                )

                show_table(display_ranking)

            else:

                st.warning(
                    "No valid diagnostic scores are available "
                    "for the selected filters."
                )

        elif score_col == "No available score":

            st.warning(
                "This dataset does not contain a supported diagnostic score column."
            )

        else:

            st.warning(
                "The selected dataset must contain both robot_id "
                "and a valid diagnostic score column."
            )

        st.markdown("---")

        section_heading(
            "Full diagnostic records",
            "Original diagnostic records after applying your selected filters."
        )

        show_table(df)


# ==========================================================
# 9. RECOVERY CENTER
# ==========================================================

elif page == "Recovery Center":

    page_header(
        "AUTONOMOUS RECOVERY CENTER",
        "Isolate faulty robots and reassign their tasks",
        "Analyze suspicious agents, isolate selected robots "
        "and evaluate the outcome of task reassignment."
    )

    histories = {}

    if not baseline_history.empty:
        histories["Baseline experiments"] = baseline_history

    if not stress_history.empty:
        histories["Stress tests"] = stress_history

    if not multi_history.empty:
        histories["Multi-fault experiments"] = multi_history

    if "live_history" in st.session_state:
        histories["Live simulation"] = st.session_state["live_history"]

    if not histories:

        st.warning("No execution history is available.")

    else:

        selected = st.selectbox(
            "Select execution dataset",
            list(histories.keys())
        )

        history = normalize_success(histories[selected])

        section_heading(
            "Diagnosis workflow",
            "Analyze the selected execution dataset before recovery."
        )

        if st.button(
            "🔍 ANALYZE EXECUTION HISTORY",
            use_container_width=True
        ):

            try:

                diagnosis = diagnose_collaboratively(history)

                explanations = explain_consensus(diagnosis)

                st.session_state["recovery_diagnosis"] = diagnosis
                st.session_state["recovery_explanations"] = explanations
                st.session_state["recovery_history"] = history
                st.session_state["recovery_dataset"] = selected

                st.session_state.pop("recovery_summary", None)
                st.session_state.pop("recovered_tasks", None)

                st.success("Diagnosis completed.")

            except Exception as error:

                st.error(f"Diagnosis error: {error}")

        if (
            "recovery_diagnosis" in st.session_state
            and st.session_state.get("recovery_dataset") == selected
        ):

            diagnosis = st.session_state["recovery_diagnosis"]

            st.markdown("---")

            section_heading(
                "Diagnostic findings",
                "Review the suspiciousness ranking before isolating robots."
            )

            show_table(diagnosis)

            with st.expander("View agent explanations"):
                show_table(st.session_state["recovery_explanations"])

            suspects = (
                diagnosis["robot_id"].tolist()
                if not diagnosis.empty and "robot_id" in diagnosis.columns
                else []
            )

            selected_suspects = st.multiselect(
                "Choose robots to isolate",
                suspects,
                default=suspects[:1],
                key="recovery_center_selection"
            )

            if st.button(
                "⚡ EXECUTE TASK RECOVERY",
                use_container_width=True
            ):

                if not selected_suspects:

                    st.error("Select at least one robot to isolate.")

                else:

                    try:

                        recovered, summary = simulate_recovery(
                            st.session_state["recovery_history"],
                            selected_suspects
                        )

                        st.session_state["recovered_tasks"] = recovered
                        st.session_state["recovery_summary"] = summary

                        st.success("Recovery simulation completed.")

                    except Exception as error:

                        st.error(f"Recovery error: {error}")

            if "recovery_summary" in st.session_state:

                st.markdown("---")

                section_heading(
                    "Recovery results",
                    "Review task reassignment and recovery performance."
                )

                show_recovery_summary(
                    st.session_state["recovery_summary"]
                )

                recovered = st.session_state["recovered_tasks"]

                show_table(recovered)

                download_csv(
                    recovered,
                    "faulttrace_recovery_report.csv",
                    "Download recovery report"
                )


# ==========================================================
# 10. EXPERIMENT ANALYTICS
# ==========================================================

elif page == "Experiment Analytics":

    page_header(
        "RESEARCH AND EXPERIMENTS",
        "Measure how accurately faults are detected",
        "Compare experimental outcomes across baseline, stress-test "
        "and multi-fault evaluation environments."
    )

    if not baseline_evaluation.empty:

        section_heading(
            "Baseline evaluation",
            "Aggregate results from the baseline experiment runs."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Top-1 Accuracy",
            format_percent(safe_mean(baseline_evaluation, "top1"))
        )

        c2.metric(
            "Top-3 Accuracy",
            format_percent(safe_mean(baseline_evaluation, "top3"))
        )

        mrr = safe_mean(baseline_evaluation, "reciprocal_rank")

        c3.metric(
            "Mean Reciprocal Rank",
            f"{mrr:.4f}" if mrr is not None and not pd.isna(mrr) else "N/A"
        )

        available_metrics = [
            col for col in [
                "top1",
                "top3",
                "reciprocal_rank"
            ]
            if col in baseline_evaluation.columns
        ]

        if available_metrics:

            selected_metric = st.selectbox(
                "Baseline metric to visualize",
                available_metrics
            )

            chart_df = baseline_evaluation.reset_index()

            fig = px.line(
                chart_df,
                x="index",
                y=selected_metric,
                markers=True,
                title=f"Baseline {selected_metric} across runs"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

    else:

        st.info("Baseline evaluation data is not available.")

    st.markdown("---")

    if not stress_summary.empty:

        section_heading(
            "Stress-test evaluation",
            "Assess how diagnostic performance changes under stress."
        )

        available = [
            col for col in [
                "top1_accuracy",
                "top3_accuracy",
                "mrr"
            ]
            if col in stress_summary.columns
        ]

        if available:

            metric = st.selectbox(
                "Stress-test evaluation metric",
                available
            )

            fig = px.bar(
                stress_summary,
                x="scenario" if "scenario" in stress_summary.columns else stress_summary.index,
                y=metric,
                color="scenario" if "scenario" in stress_summary.columns else None,
                text_auto=".3f",
                title=f"Stress-test {metric}"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        show_table(stress_summary)

    else:

        st.info("Stress-test summary data is not available.")

    st.markdown("---")

    if not multi_summary.empty:

        section_heading(
            "Multi-fault evaluation",
            "Analyze how effectively the system identifies multiple faulty agents."
        )

        available = [
            col for col in [
                "top1_hit_rate",
                "top3_hit_rate",
                "fault_recall_at_3",
                "mean_fault_rank",
                "mrr_faults"
            ]
            if col in multi_summary.columns
        ]

        if available:

            metric = st.selectbox(
                "Multi-fault evaluation metric",
                available
            )

            fig = px.bar(
                multi_summary,
                x="scenario" if "scenario" in multi_summary.columns else multi_summary.index,
                y=metric,
                color="scenario" if "scenario" in multi_summary.columns else None,
                text_auto=".3f",
                title=f"Multi-fault {metric}"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        show_table(multi_summary)

    else:

        st.info("Multi-fault summary data is not available.")


# ==========================================================
# 11. EXECUTION HISTORY
# ==========================================================

elif page == "Execution History":

    page_header(
        "EXECUTION DATA CENTER",
        "Task-level records and outcomes for every robot",
        "Inspect execution records, compare robot behaviour "
        "and investigate successful and failed operations."
    )

    histories = {}

    if not baseline_history.empty:
        histories["Baseline"] = baseline_history

    if not stress_history.empty:
        histories["Stress tests"] = stress_history

    if not multi_history.empty:
        histories["Multi-fault"] = multi_history

    if "live_history" in st.session_state:
        histories["Live simulation"] = st.session_state["live_history"]

    if not histories:

        st.warning("No execution history files were found.")

    else:

        selected = st.selectbox(
            "Select execution dataset",
            list(histories.keys())
        )

        df = normalize_success(histories[selected])

        st.markdown("---")

        section_heading(
            "Record filters",
            "Narrow down execution records to investigate specific outcomes."
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            if "robot_id" in df.columns:

                robots = sorted(
                    df["robot_id"].dropna().unique().tolist()
                )

                selected_robot = st.selectbox(
                    "Robot",
                    ["All"] + robots
                )

                if selected_robot != "All":
                    df = df[df["robot_id"] == selected_robot]

        with c2:

            if "success" in df.columns:

                outcome = st.selectbox(
                    "Outcome",
                    ["All", "Successful", "Failed"]
                )

                if outcome == "Successful":
                    df = df[df["success"] == True]

                elif outcome == "Failed":
                    df = df[df["success"] == False]

        with c3:

            st.metric(
                "Filtered records",
                len(df)
            )

        st.markdown("---")

        section_heading(
            "Execution records",
            "Detailed task-level operational history."
        )

        show_table(df)

        if not df.empty and "success" in df.columns:

            performance = fleet_performance(df)

            if not performance.empty:

                st.markdown("---")

                section_heading(
                    "Operational performance",
                    "Compare successful and failed tasks across robots."
                )

                c1, c2 = st.columns(2)

                with c1:

                    fig = px.bar(
                        performance,
                        x="robot_id",
                        y="failed_tasks",
                        color="failed_tasks",
                        title="Failed tasks per robot",
                        text_auto=True
                    )

                    st.plotly_chart(
                        style_chart(fig),
                        use_container_width=True
                    )

                with c2:

                    fig = px.bar(
                        performance,
                        x="robot_id",
                        y="success_rate",
                        color="success_rate",
                        title="Robot success rate",
                        text="success_rate"
                    )

                    fig.update_traces(
                        texttemplate="%{text:.1%}",
                        textposition="outside"
                    )

                    fig.update_yaxes(range=[0, 1.15])

                    st.plotly_chart(
                        style_chart(fig),
                        use_container_width=True
                    )

                show_table(performance)

        st.markdown("---")

        download_csv(
            df,
            "faulttrace_execution_history.csv",
            "Download filtered execution history"
        )


elif page == "API Integration":
    render_api_page(page_header, section_heading, show_table)


# ==========================================================
# 12. FOOTER
# ==========================================================

st.divider()

footer_left, footer_center, footer_right = st.columns([1, 2, 1])

with footer_left:

    st.markdown(
        "**◈ FAULTTRACE AI**",
        help="Multi-Agent Fault Diagnosis & Autonomous Recovery"
    )

with footer_center:

    st.caption(
        "Multi-Agent Fault Diagnosis & Autonomous Recovery"
    )

with footer_right:

    st.caption("VERSION 3.0")