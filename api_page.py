"""
FaultTrace AI - "API Integration" dashboard page
------------------------------------------------
Put this file in the PROJECT ROOT (next to app.py).
 
app.py needs only 3 small edits (see the instructions):
    from api_page import render_api_page
    ... add "API Integration" to the sidebar radio list ...
    elif page == "API Integration":
        render_api_page(page_header, section_heading, show_table)
"""
 
import os
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
import api_client as api
 
ROBOTS = ["R1", "R2", "R3", "R4", "R5"]
SCORE_COLUMNS = ["consensus_score", "suspiciousness", "diagnostic_score"]
 
 
def _default_url():
    try:
        return st.secrets["FAULTTRACE_API_URL"]
    except Exception:
        return os.environ.get("FAULTTRACE_API_URL", "http://127.0.0.1:8000")
 
 
def _remember(method, path, data):
    st.session_state["api_last"] = (method, path, data)
 
 
def render_api_page(page_header, section_heading, show_table):
 
    page_header(
        "API INTEGRATION",
        "Run FaultTrace through its HTTP API",
        "This page does not call Python functions directly. Every action "
        "below is a real HTTP request to the FaultTrace FastAPI server, "
        "the same way a mobile app or another system would use it."
    )
 
    # ------------------------------------------------------
    # Connection
    # ------------------------------------------------------
 
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = _default_url()
 
    section_heading(
        "Connection",
        "Address of the FaultTrace API server."
    )
 
    base_url = st.text_input("FaultTrace API address", key="api_url")
    docs_url = base_url.rstrip("/") + "/docs"
 
    left, right = st.columns([1, 3])
 
    with left:
        if st.button("CHECK CONNECTION", use_container_width=True):
            try:
                status = api.get_status(base_url)
                st.session_state["api_status"] = status
                _remember("GET", "/status", status)
            except api.ApiError as error:
                st.session_state["api_status"] = None
                st.error(str(error))
 
    with right:
        st.markdown(f"Interactive API documentation: [{docs_url}]({docs_url})")
 
    status = st.session_state.get("api_status")
 
    if status:
        c1, c2, c3 = st.columns(3)
        c1.metric("API status", str(status.get("status", "?")).upper())
        c2.metric(
            "Simulation in memory",
            "Yes" if status.get("has_simulation") else "No"
        )
        c3.metric("Tasks in memory", status.get("tasks_in_memory", 0))
 
    # ------------------------------------------------------
    # 1. Simulate
    # ------------------------------------------------------
 
    st.markdown("---")
 
    section_heading(
        "1 · Simulate",
        "POST /simulate runs the robots, injects faults and diagnoses them."
    )
 
    left, right = st.columns(2)
 
    with left:
        faulty = st.multiselect(
            "Faulty robots", ROBOTS, default=["R3"], key="api_faulty"
        )
        mode = st.selectbox(
            "Fault behaviour", ["Intermittent", "Permanent"], key="api_mode"
        )
        probability = st.slider(
            "Activation probability", 0.05, 1.0, 0.30, 0.05, key="api_prob"
        )
 
    with right:
        severity = st.slider(
            "Fault severity", 0.05, 0.95, 0.70, 0.05, key="api_sev"
        )
        tasks = st.select_slider(
            "Number of tasks",
            options=[50, 100, 150, 200, 300, 400, 500],
            value=100,
            key="api_tasks"
        )
        seed = st.number_input(
            "Random seed", min_value=1, value=42, step=1, key="api_seed"
        )
 
    if st.button("⚡ RUN SIMULATION VIA API", use_container_width=True):
 
        if not faulty:
            st.error("Select at least one faulty robot.")
 
        else:
            try:
                with st.spinner("Calling POST /simulate ..."):
                    result = api.run_simulation(base_url, {
                        "faulty_robots": faulty,
                        "fault_mode": mode,
                        "fault_probability": probability,
                        "fault_severity": severity,
                        "num_tasks": int(tasks),
                        "seed": int(seed),
                    })
 
                st.session_state["api_sim"] = result
                st.session_state["api_recovery"] = None
                _remember("POST", "/simulate", result)
                st.success("The API answered with HTTP 200.")
 
            except api.ApiError as error:
                st.error(str(error))
 
    sim = st.session_state.get("api_sim")
 
    if sim:
 
        summary = sim["summary"]
 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tasks", summary["tasks"])
        c2.metric("Successful", summary["successful"])
        c3.metric("Failed", summary["failed"])
        c4.metric("Success rate", f"{summary['success_rate']:.1%}")
 
        diagnosis = pd.DataFrame(sim["diagnosis"])
 
        score_col = next(
            (c for c in SCORE_COLUMNS if c in diagnosis.columns), None
        )
 
        left, right = st.columns([1.1, 0.9])
 
        with left:
            if score_col and "robot_id" in diagnosis.columns:
                fig = px.bar(
                    diagnosis.sort_values(score_col),
                    x=score_col,
                    y="robot_id",
                    orientation="h",
                    title="Suspicion ranking (returned by the API)"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    colorway=["#8B5CF6"],
                    height=320,
                    margin=dict(l=15, r=25, t=55, b=25),
                )
                st.plotly_chart(fig, use_container_width=True)
 
        with right:
            show_table(diagnosis)
 
        top = sim.get("top_suspect")
        injected = sim.get("config", {}).get("faulty_robots", [])
 
        if top and top in injected:
            st.success(f"Top suspect {top} matches an injected fault.")
        elif top:
            st.warning(f"Top suspect: {top}. Injected faults: {injected}")
 
        # --------------------------------------------------
        # 2. Recover
        # --------------------------------------------------
 
        st.markdown("---")
 
        section_heading(
            "2 · Recover",
            "POST /recover isolates robots and reassigns their failed tasks."
        )
 
        options = (
            diagnosis["robot_id"].tolist()
            if "robot_id" in diagnosis.columns
            else ROBOTS
        )
 
        suspects = st.multiselect(
            "Robots to isolate",
            options,
            default=options[:1],
            key="api_suspects"
        )
 
        if st.button("⚡ RUN RECOVERY VIA API", use_container_width=True):
 
            if not suspects:
                st.error("Select at least one robot.")
 
            else:
                try:
                    with st.spinner("Calling POST /recover ..."):
                        recovery = api.run_recovery(base_url, suspects)
 
                    st.session_state["api_recovery"] = recovery
                    _remember("POST", "/recover", recovery)
                    st.success("The API answered with HTTP 200.")
 
                except api.ApiError as error:
                    st.error(str(error))
 
        recovery = st.session_state.get("api_recovery")
 
        if recovery:
 
            rec = recovery.get("summary", {})
 
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Failed tasks", rec.get("failed_tasks_identified", 0))
            c2.metric("Reassigned", rec.get("tasks_reassigned", 0))
            c3.metric("Recovered", rec.get("successful_recoveries", 0))
            c4.metric("Recovery rate", f"{float(rec.get('recovery_rate', 0)):.1f}%")
 
            show_table(pd.DataFrame(recovery.get("recovered_tasks", [])))
 
    # ------------------------------------------------------
    # 3. History
    # ------------------------------------------------------
 
    st.markdown("---")
 
    section_heading(
        "3 · Execution history",
        "GET /history returns the stored task records."
    )
 
    limit = st.number_input(
        "Rows to fetch", min_value=10, max_value=1000, value=50, step=10,
        key="api_limit"
    )
 
    if st.button("LOAD HISTORY VIA API", use_container_width=True):
        try:
            data = api.get_history(base_url, limit=int(limit))
            st.session_state["api_history"] = data
            _remember("GET", "/history", data)
        except api.ApiError as error:
            st.error(str(error))
 
    history = st.session_state.get("api_history")
 
    if history:
        st.caption(
            f"Showing {len(history['records'])} of {history['total']} records."
        )
        show_table(pd.DataFrame(history["records"]))
 
    # ------------------------------------------------------
    # Raw response (proof for judges that this is real HTTP)
    # ------------------------------------------------------
 
    last = st.session_state.get("api_last")
 
    if last:
        st.markdown("---")
        with st.expander("Inspect the last raw API response"):
            st.write(f"`{last[0]} {last[1]}`")
            st.json(last[2], expanded=False)
 