"""
FaultTrace AI - FastAPI layer
-----------------------------
Put this file in the PROJECT ROOT (same level as app.py and the backend folder).
 
Run (from the project root, with the .venv activated):
    uvicorn api:app --reload
 
Then open the auto-generated docs page:
    http://127.0.0.1:8000/docs
"""
 
import asyncio
import json
from typing import List, Literal, Optional
 
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
 
from backend.simulation.engine import WarehouseSimulation
from backend.diagnosis.collaborative import (
    diagnose_collaboratively,
    explain_consensus,
)
from backend.recovery.engine import simulate_recovery
 
 
app = FastAPI(
    title="FaultTrace AI API",
    description="Multi-agent fault diagnosis and autonomous recovery for robot fleets.",
    version="1.0.0",
)
 
# Allows other websites/apps to call this API (fine for a hackathon demo).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
ROBOTS = ["R1", "R2", "R3", "R4", "R5"]
 
# The API remembers the LAST simulation in memory.
# (One shared state = fine for a demo. Restarting the server clears it.)
STATE = {
    "history": None,
    "diagnosis": None,
    "explanations": None,
    "config": None,
}
 
 
# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
 
def _json_default(obj):
    # Converts NumPy numbers (int64, float64, bool_) into normal Python values.
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
 
 
def clean(obj):
    return json.loads(json.dumps(obj, default=_json_default))
 
 
def df_to_records(df: pd.DataFrame):
    # pandas knows how to write NumPy values and NaN safely as JSON.
    return json.loads(df.to_json(orient="records"))
 
 
def normalize_success(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "success" in df.columns and df["success"].dtype != bool:
        df["success"] = (
            df["success"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    return df
 
 
def require_history() -> pd.DataFrame:
    if STATE["history"] is None:
        raise HTTPException(
            status_code=409,
            detail="No simulation has been run yet. Call POST /simulate first.",
        )
    return STATE["history"]
 
 
# ----------------------------------------------------------
# Request models (what the client must send)
# ----------------------------------------------------------
 
class SimulationRequest(BaseModel):
    faulty_robots: List[str] = Field(default_factory=lambda: ["R3"])
    fault_mode: Literal["Intermittent", "Permanent"] = "Intermittent"
    fault_probability: float = Field(0.30, ge=0.05, le=1.0)
    fault_severity: float = Field(0.70, ge=0.05, le=0.95)
    num_tasks: int = Field(100, ge=10, le=1000)
    seed: int = Field(42, ge=1)
 
 
class RecoveryRequest(BaseModel):
    suspects: List[str] = Field(..., min_length=1)
 
 
# ----------------------------------------------------------
# Endpoints
# ----------------------------------------------------------
 
@app.get("/status")
def status():
    """Is the server alive, and is there a simulation in memory?"""
    history = STATE["history"]
    return {
        "status": "ok",
        "service": "FaultTrace AI",
        "has_simulation": history is not None,
        "tasks_in_memory": 0 if history is None else int(len(history)),
        "config": STATE["config"],
    }
 
 
@app.post("/simulate")
def simulate(request: SimulationRequest):
    """Run a simulation, diagnose it, and remember the result."""
    unknown = [r for r in request.faulty_robots if r not in ROBOTS]
    if not request.faulty_robots or unknown:
        raise HTTPException(
            status_code=422,
            detail=f"faulty_robots must be a non-empty subset of {ROBOTS}.",
        )
 
    probability = 1.0 if request.fault_mode == "Permanent" else request.fault_probability
 
    try:
        simulation = WarehouseSimulation(
            num_robots=5,
            seed=request.seed,
            faulty_robot_ids=request.faulty_robots,
            fault_probability=probability,
            fault_mode=request.fault_mode,
            fault_severity=request.fault_severity,
        )
        simulation.generate_tasks(num_tasks=request.num_tasks)
        records = simulation.run()
 
        df = normalize_success(pd.DataFrame([r.to_dict() for r in records]))
        if df.empty:
            raise ValueError("The simulation generated no records.")
 
        diagnosis = diagnose_collaboratively(df)
        explanations = explain_consensus(diagnosis)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {error}")
 
    STATE["history"] = df
    STATE["diagnosis"] = diagnosis
    STATE["explanations"] = explanations
    STATE["config"] = {
        "faulty_robots": request.faulty_robots,
        "fault_mode": request.fault_mode,
        "activation_probability": probability,
        "fault_severity": request.fault_severity,
        "num_tasks": request.num_tasks,
        "seed": request.seed,
    }
 
    successful = int(df["success"].sum())
    top_suspect = None
    if not diagnosis.empty and "robot_id" in diagnosis.columns:
        top_suspect = str(diagnosis.iloc[0]["robot_id"])
 
    return {
        "config": STATE["config"],
        "summary": {
            "tasks": int(len(df)),
            "successful": successful,
            "failed": int(len(df) - successful),
            "success_rate": round(successful / len(df), 4),
        },
        "top_suspect": top_suspect,
        "diagnosis": df_to_records(diagnosis),
    }
 
 
@app.get("/diagnose")
def diagnose():
    """Return the latest diagnosis ranking and the agent explanations."""
    require_history()
    return {
        "diagnosis": df_to_records(STATE["diagnosis"]),
        "explanations": df_to_records(STATE["explanations"]),
    }
 
 
@app.post("/recover")
def recover(request: RecoveryRequest):
    """Isolate the chosen robots and simulate reassigning their failed tasks."""
    history = require_history()
 
    unknown = [r for r in request.suspects if r not in ROBOTS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown robots: {unknown}")
 
    try:
        recovered, summary = simulate_recovery(history, request.suspects)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Recovery failed: {error}")
 
    return {
        "isolated": request.suspects,
        "summary": clean(summary),
        "recovered_tasks": df_to_records(recovered),
    }
 
 
@app.get("/history")
def history(limit: int = 100, robot_id: Optional[str] = None):
    """Return execution records (optionally only one robot, and only the first N)."""
    df = require_history()
    if robot_id:
        df = df[df["robot_id"] == robot_id]
    limit = max(1, min(limit, 5000))
    return {"total": int(len(df)), "records": df_to_records(df.head(limit))}
 
 
@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket, delay: float = 0.2):
    """
    Streams the last simulation task by task, like a live feed.
    Each message includes the task record and a running tally per robot.
    """
    await websocket.accept()
    try:
        df = STATE["history"]
        if df is None:
            await websocket.send_json(
                {"type": "error", "message": "Run POST /simulate first."}
            )
            await websocket.close()
            return
 
        delay = max(0.02, min(delay, 2.0))
        records = df_to_records(df)
        total = len(records)
        tally = {}
 
        for index, record in enumerate(records, start=1):
            robot = record.get("robot_id")
            entry = tally.setdefault(robot, {"tasks": 0, "failed": 0})
            entry["tasks"] += 1
            if record.get("success") is False:
                entry["failed"] += 1
 
            await websocket.send_json(
                {
                    "type": "task",
                    "index": index,
                    "total": total,
                    "record": record,
                    "tally": tally,
                }
            )
            await asyncio.sleep(delay)
 
        await websocket.send_json({"type": "done", "tally": tally})
        await websocket.close()
 
    except WebSocketDisconnect:
        pass
 