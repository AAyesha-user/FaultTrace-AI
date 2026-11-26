
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
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

EXPERIMENTS = DATA / "experiments"
MULTI = DATA / "multi_fault"
STRESS = DATA / "stress_tests"

ROBOTS = ["R1", "R2", "R3", "R4", "R5"]

st.set_page_config(
    page_title="FaultTrace AI | Intelligence Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# PREMIUM PURPLE DESIGN
# ==========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 80% 0%, #281044 0%, transparent 40%),
        radial-gradient(ellipse at 0% 50%, #190D2B 0%, transparent 35%),
        #090612;
    color: #F4EDFF;
}

.block-container {
    max-width: 1650px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

html, body, [class*="css"] {
    font-size: 16px;
}

h1 {
    font-size: 2.5rem !important;
}

h2 {
    font-size: 2rem !important;
}

h3 {
    font-size: 1.5rem !important;
}

p, label, li {
    color: #E0D2EF !important;
    font-size: 15px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #170D27, #0B0713);
    border-right: 1px solid #75469B;
}

[data-testid="stSidebar"] * {
    color: #E9D9FA;
}

.hero {
    padding: 40px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 90% 20%, #6929A955, transparent 40%),
        linear-gradient(120deg, #28123E, #110A1D);
    border: 1px solid #7543A5;
    margin-bottom: 30px;
    box-shadow: 0 12px 40px #00000055;
}

.hero-label {
    color: #C18AFF;
    font-size: 13px;
    letter-spacing: 3px;
    font-weight: 800;
}

.hero-title {
    font-size: clamp(34px, 4vw, 48px);
    font-weight: 850;
    color: white;
    margin: 15px 0;
}

.hero-description {
    font-size: 17px;
    line-height: 1.8;
    color: #D0BDE5;
    max-width: 850px;
}

.metric-card {
    background: linear-gradient(145deg, #32194D, #160C24);
    border: 1px solid #74469A;
    border-radius: 18px;
    padding: 24px;
    min-height: 135px;
}

.metric-label {
    color: #D0B5E9;
    font-size: 13px;
    letter-spacing: 1px;
}

.metric-value {
    color: #F0D7FF;
    font-size: 34px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-sub {
    color: #BDA6D2;
    font-size: 13px;
    margin-top: 5px;
}

.feature-card {
    background: linear-gradient(145deg, #2B1740, #110B1C);
    border: 1px solid #654080;
    border-radius: 18px;
    padding: 25px;
    min-height: 145px;
    margin-bottom: 15px;
}

.feature-icon {
    font-size: 28px;
}

.feature-title {
    color: #F0D9FF;
    font-size: 19px;
    font-weight: 750;
}

.feature-description {
    color: #C2B0D4;
    font-size: 15px;
    line-height: 1.8;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(110deg, #792BD5, #C13EE6);
    color: white !important;
    border: 1px solid #D9A4FF;
    border-radius: 12px;
    min-height: 48px;
    font-size: 16px;
    font-weight: 750;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(110deg, #A54AFF, #E04FEA);
    border-color: white;
}

[data-baseweb="select"] > div {
    background-color: #21132F;
    border-color: #75469B;
}

.stTextInput input,
.stNumberInput input {
    background: #21132F;
    color: white;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #75469B;
    border-radius: 14px;
}

[data-testid="stDataFrame"] * {
    font-size: 14px;
}

[data-testid="stExpander"] {
    background: #160D23;
    border: 1px solid #563474;
    border-radius: 13px;
}

[data-testid="stMetric"] {
    background: #1C102B;
    border: 1px solid #563474;
    border-radius: 12px;
    padding: 15px;
}

hr {
    border-color: #553078;
}

.small-caption {
    color: #BD8FE0;
    font-size: 13px;
    letter-spacing: 2px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def load_csv(path):

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as error:
            st.warning(f"Could not load {path.name}: {error}")

    return pd.DataFrame()


def hero(title, subtitle, section="FAULTTRACE INTELLIGENCE CENTER"):

    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">{section}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-description">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, subtitle="Research metric"):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def feature_card(icon, title, description):

    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def style_chart(fig):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            color="#E6D7F5",
            size=14
        ),
        margin=dict(l=20, r=20, t=55, b=25),
        colorway=[
            "#A855F7",
            "#D946EF",
            "#7C3AED",
            "#E879F9",
            "#C084FC"
        ]
    )

    return fig


def normalize_success(df):

    df = df.copy()

    if "success" in df.columns:

        if df["success"].dtype != bool:

            df["success"] = (
                df["success"]
                .astype(str)
                .str.lower()
                .map({
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                })
            )

    return df


def fleet_performance(df):

    df = normalize_success(df)

    if df.empty or "robot_id" not in df.columns:
        return pd.DataFrame()

    if "success" not in df.columns:
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


# ==========================================================
# DATA LOADING
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
# SIDEBAR
# ==========================================================

with st.sidebar:


st.markdown("""
<div style="padding:15px 0 25px 0">
<div style="font-size:25px; font-weight:800; color:#D9A4FF; letter-spacing:1px;">
◈ FAULTTRACE
</div>
<div style="color:#BD8FE0; font-size:12px; letter-spacing:2px; margin-top:7px;">
AUTONOMOUS DIAGNOSTICS
</div>
</div>
""", unsafe_allow_html=True)


    st.markdown(
        '<div class="small-caption">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Fleet Overview",
            "Live Simulator",
            "Diagnosis Explorer",
            "Recovery Center",
            "Experiment Analytics",
            "Execution History"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div style="font-size:13px;color:#BDA6D2">
        🟢 DIAGNOSTIC ENGINE READY
    </div>

    <div style="
        margin-top:35px;
        color:#927FA8;
        font-size:12px;
        line-height:1.8;">
        FAULTTRACE AI<br>
        Research-driven multi-agent fault diagnosis<br>
        Version 3.0
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# FLEET OVERVIEW
# ==========================================================

if page == "Fleet Overview":

    hero(
        "Intelligence at the edge.",
        "Monitor robotic operations, identify unreliable agents "
        "and understand system-wide diagnostic performance."
    )

    st.markdown("### System Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Robot fleet", "05", "Simulated robotic agents")

    with c2:
        metric_card(
            "Baseline runs",
            len(baseline_evaluation),
            "Single-fault experiments"
        )

    with c3:
        metric_card(
            "Stress scenarios",
            len(stress_summary),
            "Experimental conditions"
        )

    with c4:
        metric_card(
            "Multi-fault scenarios",
            len(multi_summary),
            "Multiple faulty agents"
        )

    st.markdown("### Intelligence Modules")

    c1, c2 = st.columns(2)

    with c1:

        feature_card(
            "◈",
            "Baseline Diagnosis",
            "Single-fault identification using Ochiai suspiciousness."
        )

        feature_card(
            "◉",
            "Collaborative Diagnosis",
            "Multiple diagnostic agents evaluate robot reliability."
        )

        feature_card(
            "⌁",
            "Recovery Engine",
            "Failed tasks are identified and reassigned."
        )

    with c2:

        feature_card(
            "⬡",
            "Multi-Fault Diagnosis",
            "Investigate systems containing multiple faulty agents."
        )

        feature_card(
            "⚡",
            "Stress Testing",
            "Evaluate diagnostic reliability under different conditions."
        )

        feature_card(
            "◎",
            "Performance Evaluation",
            "Measure Top-1, Top-3 and ranking performance."
        )

    st.markdown("### Research Performance")

    if not stress_summary.empty:

        fig = px.bar(
            stress_summary,
            x="scenario",
            y="top1_accuracy",
            color="scenario",
            title="Stress Test Diagnostic Accuracy",
            text_auto=".1%"
        )

        fig.update_yaxes(range=[0, 1])

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    if not multi_summary.empty:

        st.markdown("### Multi-Fault Results")

        st.dataframe(
            multi_summary,
            use_container_width=True,
            hide_index=True
        )

    if not baseline_evaluation.empty:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Top-1 Accuracy",
            f"{baseline_evaluation['top1'].mean():.2%}"
        )

        c2.metric(
            "Top-3 Accuracy",
            f"{baseline_evaluation['top3'].mean():.2%}"
        )

        c3.metric(
            "Mean Reciprocal Rank",
            f"{baseline_evaluation['reciprocal_rank'].mean():.4f}"
        )


# ==========================================================
# LIVE SIMULATOR - JUDGE CONTROL PANEL
# ==========================================================

elif page == "Live Simulator":

    hero(
        "Fault Injection Lab.",
        "Configure robotic failures, execute controlled simulations "
        "and observe the response of the diagnostic intelligence engine.",
        "INTERACTIVE JUDGE CONTROL ENVIRONMENT"
    )

    st.markdown("### ◈ Judge Control Panel")

    st.markdown("#### 1. Select Faulty Robots")

    faulty_robots = st.multiselect(
        "Choose one or multiple robots",
        ROBOTS,
        default=["R3"],
        help="Select the robots that will receive injected faults."
    )

    st.markdown("#### 2. Configure Fault Behaviour")

    c1, c2 = st.columns(2)

    with c1:

        fault_mode = st.selectbox(
            "Fault Type",
            [
                "Intermittent",
                "Permanent"
            ],
            help="Intermittent faults activate probabilistically. "
                 "Permanent faults activate on every task assigned "
                 "to a selected robot."
        )

        fault_probability = st.slider(
            "Fault Activation Probability",
            0.05,
            1.0,
            0.30,
            0.05
        )

    with c2:

        fault_severity = st.slider(
            "Fault Severity",
            0.05,
            0.95,
            0.70,
            0.05,
            help="Controls the additional failure probability "
                 "when faulty behaviour activates."
        )

        num_tasks = st.slider(
            "Simulation Tasks",
            50,
            500,
            100,
            50
        )

    seed = st.number_input(
        "Simulation Seed",
        min_value=1,
        value=42
    )

    if fault_mode == "Permanent":

        effective_probability = 1.0

        st.info(
            "Permanent mode selected: every task assigned to "
            "a selected faulty robot activates faulty behaviour."
        )

    else:

        effective_probability = fault_probability

        st.info(
            f"Intermittent mode selected: "
            f"{fault_probability:.0%} activation probability."
        )

    st.caption(
        "Severity controls the additional failure probability "
        "when a fault activates. Normal operational failures "
        "remain possible."
    )

    st.divider()

    st.markdown("#### 3. Execute Simulation")

    if st.button(
        "⚡ RUN FAULT INJECTION SIMULATION",
        use_container_width=True
    ):

        if not faulty_robots:

            st.error("Please select at least one faulty robot.")

        else:

            try:

                with st.spinner(
                    "Simulating robotic operations and running "
                    "collaborative diagnosis..."
                ):

                    simulation = WarehouseSimulation(
                        num_robots=5,
                        seed=int(seed),
                        faulty_robot_ids=faulty_robots,
                        fault_probability=effective_probability,
                        fault_mode=fault_mode,
                        fault_severity=fault_severity
                    )

                    simulation.generate_tasks(
                        num_tasks=num_tasks
                    )

                    records = simulation.run()

                    df = pd.DataFrame(
                        [record.to_dict() for record in records]
                    )

                    df = normalize_success(df)

                    diagnosis = diagnose_collaboratively(df)

                    explanations = explain_consensus(diagnosis)

                    st.session_state["live_history"] = df

                    st.session_state["live_diagnosis"] = diagnosis

                    st.session_state["live_explanations"] = explanations

                    st.session_state["live_faults"] = faulty_robots

                    st.session_state["live_mode"] = fault_mode

                    st.session_state["live_severity"] = fault_severity

                    st.session_state["live_probability"] = effective_probability

                    st.session_state["live_recovery"] = None

                st.success("Simulation and collaborative diagnosis completed!")

            except Exception as error:

                st.error(f"Simulation error: {error}")

    # ------------------------------------------------------
    # SIMULATION RESULTS
    # ------------------------------------------------------

    if "live_history" in st.session_state:

        df = st.session_state["live_history"]

        diagnosis = st.session_state["live_diagnosis"]

        actual_faults = st.session_state["live_faults"]

        st.divider()

        st.markdown("## Simulation Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Tasks Executed", len(df))

        c2.metric(
            "Successful Tasks",
            int(df["success"].sum())
        )

        c3.metric(
            "Failed Tasks",
            int((~df["success"]).sum())
        )

        c4.metric(
            "Injected Faulty Robots",
            len(actual_faults)
        )

        st.markdown("### Simulation Configuration")

        st.write(
            f"**Faulty robots:** {', '.join(actual_faults)}"
        )

        st.write(
            f"**Fault mode:** {st.session_state['live_mode']}"
        )

        st.write(
            f"**Fault probability:** "
            f"{st.session_state['live_probability']:.0%}"
        )

        st.write(
            f"**Fault severity:** "
            f"{st.session_state['live_severity']:.0%}"
        )

        st.divider()

        st.markdown("## Collaborative AI Diagnosis")

        c1, c2 = st.columns(2)

        with c1:

            fig = px.bar(
                diagnosis,
                x="robot_id",
                y="consensus_score",
                color="consensus_score",
                title="Collaborative Diagnostic Scores",
                text_auto=".3f"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        with c2:

            st.markdown("### Diagnostic Ranking")

            st.dataframe(
                diagnosis,
                use_container_width=True,
                hide_index=True
            )

        if not diagnosis.empty:

            predicted = diagnosis.iloc[0]["robot_id"]

            if predicted in actual_faults:

                st.success(
                    f"Correct identification: {predicted} "
                    "was ranked first."
                )

            else:

                st.warning(
                    f"First-ranked robot: {predicted}. "
                    f"Injected faults: {actual_faults}"
                )

        with st.expander("View Diagnostic Agent Explanations"):

            st.dataframe(
                st.session_state["live_explanations"],
                use_container_width=True,
                hide_index=True
            )

        with st.expander("View Complete Execution History"):

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        # --------------------------------------------------
        # AUTONOMOUS RECOVERY
        # --------------------------------------------------

        st.divider()

        st.markdown("## Autonomous Recovery")

        st.write(
            "Isolate suspicious robots and reassign their failed "
            "tasks to available healthy robots."
        )

        available_robots = (
            diagnosis["robot_id"].tolist()
            if not diagnosis.empty
            else []
        )

        selected_suspects = st.multiselect(
            "Select robots to isolate",
            available_robots,
            default=[
                diagnosis.iloc[0]["robot_id"]
            ] if not diagnosis.empty else [],
            key="live_recovery_selection"
        )

        if st.button(
            "⚡ EXECUTE AUTONOMOUS RECOVERY",
            use_container_width=True
        ):

            if not selected_suspects:

                st.error("Select at least one robot to isolate.")

            else:

                try:

                    with st.spinner(
                        "Reassigning failed tasks..."
                    ):

                        recovered, summary = simulate_recovery(
                            df,
                            selected_suspects
                        )

                        st.session_state["live_recovery"] = recovered

                        st.session_state["live_recovery_summary"] = summary

                        st.session_state["live_isolated"] = selected_suspects

                    st.success("Recovery simulation completed!")

                except Exception as error:

                    st.error(f"Recovery failed: {error}")

        if st.session_state.get("live_recovery") is not None:

            recovered = st.session_state["live_recovery"]

            summary = st.session_state["live_recovery_summary"]

            st.markdown("### Recovery Performance")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Failed Tasks Identified",
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

            c4.metric(
                "Recovery Rate",
                f"{summary.get('recovery_rate', 0):.1f}%"
            )

            st.write(
                "**Isolated robots:** "
                + ", ".join(st.session_state["live_isolated"])
            )

            st.markdown("### Reassigned Task Records")

            st.dataframe(
                recovered,
                use_container_width=True,
                hide_index=True
            )

            csv = recovered.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Recovery Report",
                data=csv,
                file_name="faulttrace_recovery_report.csv",
                mime="text/csv"
            )

    # ------------------------------------------------------
    # FLEET PERFORMANCE
    # ------------------------------------------------------

    st.divider()

    st.markdown("### Historical Robot Fleet Performance")

    if not baseline_history.empty:

        performance = fleet_performance(baseline_history)

        if not performance.empty:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Robot Tasks",
                int(performance["total_tasks"].sum())
            )

            c2.metric(
                "Fleet Success Rate",
                f"{performance['successful_tasks'].sum() / performance['total_tasks'].sum():.2%}"
            )

            c3.metric(
                "Fleet Failed Tasks",
                int(performance["failed_tasks"].sum())
            )

            fig = px.bar(
                performance,
                x="robot_id",
                y="success_rate",
                color="success_rate",
                title="Historical Robot Success Rate",
                text_auto=".1%"
            )

            fig.update_yaxes(range=[0, 1.15])

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

    else:

        st.warning("No baseline execution history available.")


# ==========================================================
# DIAGNOSIS EXPLORER
# ==========================================================

elif page == "Diagnosis Explorer":

    hero(
        "Diagnostic Intelligence.",
        "Investigate suspicious robotic agents and explore "
        "the evidence behind fault rankings.",
        "AI DIAGNOSTIC CENTER"
    )

    datasets = {}

    if not baseline_rankings.empty:
        datasets["Baseline"] = baseline_rankings

    if not stress_rankings.empty:
        datasets["Stress Tests"] = stress_rankings

    if not multi_rankings.empty:
        datasets["Multi-Fault"] = multi_rankings

    if "live_diagnosis" in st.session_state:
        datasets["Live Collaborative Diagnosis"] = (
            st.session_state["live_diagnosis"]
        )

    if not datasets:

        st.warning("No diagnostic ranking files found.")

    else:

        dataset_name = st.selectbox(
            "Diagnostic Dataset",
            list(datasets.keys())
        )

        df = datasets[dataset_name].copy()

        if "scenario" in df.columns:

            scenarios = sorted(
                df["scenario"].dropna().unique().tolist()
            )

            if scenarios:

                scenario = st.selectbox(
                    "Select Scenario",
                    scenarios
                )

                df = df[df["scenario"] == scenario]

        if "run_id" in df.columns:

            run_ids = sorted(
                df["run_id"].dropna().unique().tolist()
            )

            if run_ids:

                run_id = st.selectbox(
                    "Select Experiment Run",
                    run_ids
                )

                df = df[df["run_id"] == run_id]

        score_column = next(
            (
                col for col in [
                    "consensus_score",
                    "suspiciousness",
                    "diagnostic_score"
                ]
                if col in df.columns
            ),
            None
        )

        if score_column:

            df = df.sort_values(
                score_column,
                ascending=False
            )

            fig = px.bar(
                df,
                x="robot_id",
                y=score_column,
                color=score_column,
                title="Robot Fault Ranking",
                text_auto=".4f"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# RECOVERY CENTER
# ==========================================================

elif page == "Recovery Center":

    hero(
        "Autonomous Recovery.",
        "Identify failed operations, isolate suspicious robots "
        "and reassign failed tasks to healthy agents.",
        "FAULT RECOVERY AND TASK REALLOCATION"
    )

    histories = {}

    if not baseline_history.empty:
        histories["Baseline Experiments"] = baseline_history

    if not stress_history.empty:
        histories["Stress Tests"] = stress_history

    if not multi_history.empty:
        histories["Multi-Fault Experiments"] = multi_history

    if "live_history" in st.session_state:
        histories["Live Simulation"] = st.session_state["live_history"]

    if not histories:

        st.warning("No execution histories available.")

    else:

        selected = st.selectbox(
            "Choose execution history",
            list(histories.keys())
        )

        history = normalize_success(
            histories[selected]
        )

        if st.button(
            "🔍 ANALYZE ROBOT FLEET",
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

                st.success("Collaborative diagnosis completed.")

            except Exception as error:

                st.error(f"Diagnosis failed: {error}")

        if "recovery_diagnosis" in st.session_state:

            diagnosis = st.session_state["recovery_diagnosis"]

            st.markdown("### Diagnostic Results")

            st.dataframe(
                diagnosis,
                use_container_width=True,
                hide_index=True
            )

            with st.expander("View Agent Explanations"):

                st.dataframe(
                    st.session_state["recovery_explanations"],
                    use_container_width=True,
                    hide_index=True
                )

            suspects = diagnosis["robot_id"].tolist()

            selected_suspects = st.multiselect(
                "Robots to isolate",
                suspects,
                default=suspects[:1],
                key="recovery_center_selection"
            )

            if st.button(
                "⚡ EXECUTE AUTONOMOUS RECOVERY",
                use_container_width=True
            ):

                if not selected_suspects:

                    st.error("Select at least one suspicious robot.")

                else:

                    try:

                        recovered, summary = simulate_recovery(
                            st.session_state["recovery_history"],
                            selected_suspects
                        )

                        st.session_state["recovered_tasks"] = recovered

                        st.session_state["recovery_summary"] = summary

                        st.success("Recovery completed!")

                    except Exception as error:

                        st.error(f"Recovery failed: {error}")

            if "recovery_summary" in st.session_state:

                summary = st.session_state["recovery_summary"]

                recovered = st.session_state["recovered_tasks"]

                st.markdown("### Recovery Performance")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Failed Tasks Identified",
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

                st.progress(
                    max(0.0, min(rate / 100, 1.0))
                )

                st.dataframe(
                    recovered,
                    use_container_width=True,
                    hide_index=True
                )

                csv = recovered.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Recovery Report",
                    data=csv,
                    file_name="faulttrace_recovery_report.csv",
                    mime="text/csv"
                )


# ==========================================================
# EXPERIMENT ANALYTICS
# ==========================================================

elif page == "Experiment Analytics":

    hero(
        "Research & Performance.",
        "Explore diagnostic accuracy, experimental results "
        "and resilience across different fault conditions.",
        "EXPERIMENT ANALYTICS"
    )

    if not stress_summary.empty:

        st.markdown("### Stress Test Performance")

        metric = st.selectbox(
            "Stress Test Metric",
            [
                "top1_accuracy",
                "top3_accuracy",
                "mrr"
            ]
        )

        if metric in stress_summary.columns:

            fig = px.bar(
                stress_summary,
                x="scenario",
                y=metric,
                color="scenario",
                title=f"Stress Test — {metric}",
                text_auto=".3f"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

            st.dataframe(
                stress_summary,
                use_container_width=True,
                hide_index=True
            )

    if not multi_summary.empty:

        st.markdown("### Multi-Fault Performance")

        metric_multi = st.selectbox(
            "Multi-Fault Metric",
            [
                "top1_hit_rate",
                "top3_hit_rate",
                "fault_recall_at_3",
                "mean_fault_rank",
                "mrr_faults"
            ]
        )

        if metric_multi in multi_summary.columns:

            fig = px.bar(
                multi_summary,
                x="scenario",
                y=metric_multi,
                color="scenario",
                title=f"Multi-Fault — {metric_multi}",
                text_auto=".3f"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

            st.dataframe(
                multi_summary,
                use_container_width=True,
                hide_index=True
            )

    if not baseline_evaluation.empty:

        st.markdown("### Baseline Performance")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Top-1 Accuracy",
            f"{baseline_evaluation['top1'].mean():.2%}"
        )

        c2.metric(
            "Top-3 Accuracy",
            f"{baseline_evaluation['top3'].mean():.2%}"
        )

        c3.metric(
            "Mean Reciprocal Rank",
            f"{baseline_evaluation['reciprocal_rank'].mean():.4f}"
        )


# ==========================================================
# EXECUTION HISTORY
# ==========================================================

elif page == "Execution History":

    hero(
        "Operational Intelligence.",
        "Inspect execution records, task outcomes and "
        "robot-level performance.",
        "EXECUTION DATA CENTER"
    )

    histories = {}

    if not baseline_history.empty:
        histories["Baseline"] = baseline_history

    if not stress_history.empty:
        histories["Stress Tests"] = stress_history

    if not multi_history.empty:
        histories["Multi-Fault"] = multi_history

    if "live_history" in st.session_state:
        histories["Live Simulation"] = st.session_state["live_history"]

    if not histories:

        st.warning("No execution history files found.")

    else:

        selected_history = st.selectbox(
            "Select Execution Dataset",
            list(histories.keys())
        )

        df = normalize_success(
            histories[selected_history]
        )

        c1, c2 = st.columns(2)

        with c1:

            if "robot_id" in df.columns:

                robot_ids = sorted(
                    df["robot_id"].dropna().unique().tolist()
                )

                robot = st.selectbox(
                    "Filter Robot",
                    ["All"] + robot_ids
                )

                if robot != "All":
                    df = df[df["robot_id"] == robot]

        with c2:

            if "success" in df.columns:

                outcome = st.selectbox(
                    "Task Outcome",
                    ["All", "Successful", "Failed"]
                )

                if outcome == "Successful":
                    df = df[df["success"] == True]

                elif outcome == "Failed":
                    df = df[df["success"] == False]

        st.markdown("### Execution Records")

        st.write(f"Records displayed: {len(df)}")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        performance = fleet_performance(df)

        if not performance.empty:

            st.markdown("### Robot Task Performance")

            c1, c2 = st.columns(2)

            with c1:

                fig = px.bar(
                    performance,
                    x="robot_id",
                    y="failed_tasks",
                    color="robot_id",
                    title="Failed Tasks per Robot",
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
                    color="robot_id",
                    title="Robot Success Rate",
                    text_auto=".1%"
                )

                fig.update_yaxes(range=[0, 1])

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

            st.dataframe(
                performance,
                use_container_width=True,
                hide_index=True
            )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Execution Records",
            data=csv,
            file_name="faulttrace_execution_history.csv",
            mime="text/csv"
        )
