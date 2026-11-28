from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from backend.simulation.engine import WarehouseSimulation
from backend.diagnosis.suspiciousness import calculate_ochiai

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

MULTI = DATA / "multi_fault"
STRESS = DATA / "stress_tests"
EXPERIMENTS = DATA / "experiments"

st.set_page_config(
    page_title="FaultTrace AI",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


multi_summary = load_csv(MULTI / "scenario_summary.csv")
stress_summary = load_csv(STRESS / "scenario_summary.csv")
baseline_evaluation = load_csv(
    EXPERIMENTS / "evaluation_per_run.csv"
)

multi_rankings = load_csv(
    MULTI / "diagnostic_rankings.csv"
)

stress_rankings = load_csv(
    STRESS / "diagnostic_rankings.csv"
)

multi_history = load_csv(
    MULTI / "execution_history.csv"
)

stress_history = load_csv(
    STRESS / "execution_history.csv"
)

baseline_rankings = load_csv(
    EXPERIMENTS / "diagnostic_rankings.csv"
)

baseline_history = load_csv(
    EXPERIMENTS / "execution_history.csv"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("FaultTrace AI")

st.sidebar.caption(
    "Research-driven multi-agent fault diagnosis"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Fleet Overview",
        "Live Simulator",
        "Diagnosis Explorer",
        "Experiment Analytics",
        "Execution History",
    ],
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 FaultTrace AI")

st.markdown(
    "### Intelligent intermittent fault diagnosis "
    "for multi-agent robotic systems"
)

st.markdown(
    "Simulate robotic operations, investigate failures, "
    "and evaluate diagnostic performance."
)

st.divider()


# --------------------------------------------------
# Fleet Overview
# --------------------------------------------------

if page == "Fleet Overview":

    st.subheader("Research Overview")

    total_runs = 0

    if not baseline_evaluation.empty:
        total_runs += len(baseline_evaluation)

    if not stress_summary.empty:
        total_runs += 150

    if not multi_summary.empty:
        total_runs += 60

    col1, col2, col3 = st.columns(3)

    col1.metric("Robots in baseline", "5")
    col2.metric("Baseline experiment runs", "30")
    col3.metric("Multi-fault experiment runs", "60")

    st.subheader("Research Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Baseline Diagnosis")
        st.write(
            "Single-fault experiments using "
            "Ochiai suspiciousness ranking."
        )

        st.markdown("#### Stress Testing")
        st.write(
            "Five scenarios examining diagnostic "
            "performance under different conditions."
        )

    with col2:
        st.markdown("#### Multi-Fault Diagnosis")
        st.write(
            "Experiments involving two and three "
            "simultaneously faulty robots."
        )

        st.markdown("#### Evaluation Metrics")
        st.write(
            "Top-1, Top-3, fault recall, rank and "
            "reciprocal-rank measurements."
        )

    st.subheader("Experiment Results")

    if not stress_summary.empty:

        st.markdown("#### Stress Test Scenarios")

        st.dataframe(
            stress_summary,
            use_container_width=True,
        )

    if not multi_summary.empty:

        st.markdown("#### Multi-Fault Scenarios")

        st.dataframe(
            multi_summary,
            use_container_width=True,
        )

    if not baseline_evaluation.empty:

        st.markdown("#### Baseline Evaluation")

        st.metric(
            "Baseline Top-1 Accuracy",
            f"{baseline_evaluation['top1'].mean():.2%}",
        )

        st.metric(
            "Baseline Top-3 Accuracy",
            f"{baseline_evaluation['top3'].mean():.2%}",
        )


# --------------------------------------------------
# Diagnosis Explorer
# --------------------------------------------------

elif page == "Diagnosis Explorer":

    st.subheader("Diagnosis Explorer")

    datasets = {}

    if not baseline_rankings.empty:
        datasets["Baseline"] = baseline_rankings

    if not stress_rankings.empty:
        datasets["Stress Tests"] = stress_rankings

    if not multi_rankings.empty:
        datasets["Multi-Fault"] = multi_rankings

    if not datasets:

        st.warning("No diagnostic ranking files found.")

    else:

        dataset_name = st.selectbox(
            "Select diagnostic dataset",
            list(datasets.keys()),
        )

        df = datasets[dataset_name].copy()

        if "scenario" in df.columns:

            scenarios = sorted(
                df["scenario"].dropna().unique().tolist()
            )

            scenario = st.selectbox(
                "Scenario",
                scenarios,
            )

            df = df[df["scenario"] == scenario]

        if "run_id" in df.columns:

            run_ids = sorted(
                df["run_id"].dropna().unique().tolist()
            )

            run_id = st.selectbox(
                "Experiment run",
                run_ids,
            )

            df = df[df["run_id"] == run_id]

        if "suspiciousness" in df.columns:

            df = df.sort_values(
                "suspiciousness",
                ascending=False,
            )

            st.markdown("### Robot Suspiciousness Ranking")

            fig = px.bar(
                df,
                x="robot_id",
                y="suspiciousness",
                color="suspiciousness",
                title="Diagnostic Ranking",
                labels={
                    "robot_id": "Robot",
                    "suspiciousness": "Suspiciousness",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                df,
                use_container_width=True,
            )


# --------------------------------------------------
# Experiment Analytics
# --------------------------------------------------

elif page == "Experiment Analytics":

    st.subheader("Experiment Analytics")

    if not stress_summary.empty:

        st.markdown("### Stress Test Performance")

        metric = st.selectbox(
            "Select metric",
            [
                "top1_accuracy",
                "top3_accuracy",
                "mrr",
            ],
        )

        if metric in stress_summary.columns:

            fig = px.bar(
                stress_summary,
                x="scenario",
                y=metric,
                title=f"Stress Test: {metric}",
                labels={
                    "scenario": "Scenario",
                    metric: metric,
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    if not multi_summary.empty:

        st.markdown("### Multi-Fault Performance")

        metric_multi = st.selectbox(
            "Multi-fault metric",
            [
                "top1_hit_rate",
                "top3_hit_rate",
                "fault_recall_at_3",
                "mean_fault_rank",
                "mrr_faults",
            ],
        )

        if metric_multi in multi_summary.columns:

            fig = px.bar(
                multi_summary,
                x="scenario",
                y=metric_multi,
                title=f"Multi-Fault: {metric_multi}",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    if not baseline_evaluation.empty:

        st.markdown("### Baseline Performance")

        st.write(
            "Top-1 accuracy:",
            f"{baseline_evaluation['top1'].mean():.2%}",
        )

        st.write(
            "Top-3 accuracy:",
            f"{baseline_evaluation['top3'].mean():.2%}",
        )

        st.write(
            "MRR:",
            f"{baseline_evaluation['reciprocal_rank'].mean():.4f}",
        )


# --------------------------------------------------
# Live Fault Injection Simulator
# --------------------------------------------------

elif page == "Live Simulator":

    st.subheader("Interactive Robot Fault Simulator")

    st.write(
        "Inject faults into robotic agents and observe "
        "how FaultTrace AI identifies unreliable robots."
    )

    st.markdown("### Simulation Configuration")

    robot_count = 5

    faulty_robots = st.multiselect(
        "Select robots to inject faults into",
        ["R1", "R2", "R3", "R4", "R5"],
        default=["R3"],
    )

    fault_probability = st.slider(
        "Fault activation probability",
        min_value=0.05,
        max_value=1.0,
        value=0.30,
        step=0.05,
    )

    num_tasks = st.slider(
        "Number of tasks",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
    )

    seed = st.number_input(
        "Simulation seed",
        min_value=1,
        value=42,
        step=1,
    )

    if st.button("Run Fault Injection Experiment"):

        simulation = WarehouseSimulation(
            num_robots=robot_count,
            seed=int(seed),
            faulty_robot_ids=faulty_robots,
            fault_probability=fault_probability,
        )

        simulation.generate_tasks(num_tasks=num_tasks)

        records = simulation.run()

        df = pd.DataFrame(
            [record.to_dict() for record in records]
        )

        diagnosis = calculate_ochiai(df)

        st.session_state["live_history"] = df
        st.session_state["live_diagnosis"] = diagnosis
        st.session_state["live_faults"] = faulty_robots

    if "live_diagnosis" in st.session_state:

        df = st.session_state["live_history"]
        diagnosis = st.session_state["live_diagnosis"]
        actual_faults = st.session_state["live_faults"]

        st.divider()

        st.markdown("### Simulation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Tasks executed", len(df))

        col2.metric(
            "Successful tasks",
            int(df["success"].sum()),
        )

        col3.metric(
            "Failed tasks",
            int((~df["success"]).sum()),
        )

        st.markdown("### AI Diagnosis")

        diagnosis = diagnosis.sort_values(
            "suspiciousness",
            ascending=False,
        )

        st.dataframe(
            diagnosis,
            use_container_width=True,
        )

        fig = px.bar(
            diagnosis,
            x="robot_id",
            y="suspiciousness",
            color="suspiciousness",
            title="Robot Fault Suspiciousness",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        if not diagnosis.empty:

            predicted_robot = diagnosis.iloc[0]["robot_id"]

            if predicted_robot in actual_faults:

                st.success(
                    f"Correct! AI ranked {predicted_robot} "
                    "as the most suspicious robot."
                )

            else:

                st.warning(
                    f"AI ranked {predicted_robot} first. "
                    "The actual injected faults were "
                    f"{actual_faults}."
                )

        st.markdown("### Actual Fault Configuration")

        st.write("Injected faulty robots:", actual_faults)

        st.markdown("### Robot Execution History")

        st.dataframe(
            df,
            use_container_width=True,
        )


# --------------------------------------------------
# Execution History
# --------------------------------------------------

elif page == "Execution History":

    st.subheader("Execution History")

    histories = {}

    if not baseline_history.empty:
        histories["Baseline"] = baseline_history

    if not stress_history.empty:
        histories["Stress Tests"] = stress_history

    if not multi_history.empty:
        histories["Multi-Fault"] = multi_history

    if not histories:

        st.warning("No execution histories found.")

    else:

        selected_history = st.selectbox(
            "Select execution history",
            list(histories.keys()),
        )

        df = histories[selected_history].copy()

        if "robot_id" in df.columns:

            robot_ids = sorted(
                df["robot_id"].unique().tolist()
            )

            robot = st.selectbox(
                "Filter by robot",
                ["All"] + robot_ids,
            )

            if robot != "All":
                df = df[df["robot_id"] == robot]

        if "success" in df.columns:

            success_filter = st.selectbox(
                "Task outcome",
                ["All", "Successful", "Failed"],
            )

            if success_filter == "Successful":
                df = df[df["success"] == True]

            elif success_filter == "Failed":
                df = df[df["success"] == False]

        st.write(f"Records displayed: {len(df)}")

        st.dataframe(
            df,
            use_container_width=True,
        )

        if "robot_id" in df.columns and "success" in df.columns:

            performance = df.groupby("robot_id").agg(
                total_tasks=("success", "count"),
                successful_tasks=("success", "sum"),
            ).reset_index()

            performance["failed_tasks"] = (
                performance["total_tasks"]
                - performance["successful_tasks"]
            )

            st.markdown("### Robot Task Performance")

            fig = px.bar(
                performance,
                x="robot_id",
                y="failed_tasks",
                title="Failed Tasks per Robot",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )