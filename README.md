# FaultTrace AI

### Intelligent Fault Diagnosis and Recovery in Multi-Agent Systems

**FaultTrace AI** is an interactive software prototype that identifies suspicious robot behavior, analyzes system failures, and simulates recovery in a multi-agent warehouse environment.

Built for the **CYE 2026 Innovation Sprint**, the project explores how intelligent diagnostic techniques can help identify unreliable agents and improve the reliability of multi-agent systems.

## Key Features

- **Multi-Agent Warehouse Simulation:** Simulates a fleet of robots performing different warehouse tasks.
- **Configurable Fault Injection:** Introduces intermittent, permanent, and multiple-agent faults to test system behavior.
- **Execution Monitoring:** Records task outcomes and robot performance for diagnostic analysis.
- **Fault Diagnosis:** Uses spectrum-based suspiciousness analysis and an MRSD-inspired diagnostic approach to rank potentially faulty robots.
- **Collaborative Diagnosis:** Combines different diagnostic perspectives to analyze and compare suspicious behavior.
- **Recovery Simulation:** Reassigns failed tasks to alternative robots and simulates their retry outcomes.
- **Experiment Analytics:** Evaluates diagnostic performance across repeated runs, multiple-fault scenarios, and stress tests.
- **Interactive Dashboard and API:** Provides a Streamlit interface and FastAPI endpoints for interacting with the system.

## Overview

In multi-agent systems, the failure of a single unreliable agent can affect task completion and overall system performance. Intermittent faults are particularly difficult to identify because an agent may work correctly during some executions and fail during others.

FaultTrace AI explores this problem through a simulation-based workflow:

1. **Simulate:** Robots perform different tasks in a warehouse environment.
2. **Inject faults:** Different fault scenarios are introduced to observe their effects.
3. **Monitor:** Task outcomes and robot performance are recorded.
4. **Diagnose:** Diagnostic methods analyze execution data and rank potentially faulty agents.
5. **Recover:** Failed tasks are reassigned to other robots, and retry outcomes are recorded.
6. **Evaluate:** Experiments measure diagnostic performance under different conditions.

## Application

The interactive dashboard provides access to:

- Fleet Overview
- Live Simulator
- Diagnosis Explorer
- Recovery Center
- Experiment Analytics
- Execution History
- API Integration

## Technology Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Backend API:** FastAPI
- **Data Processing:** Pandas and NumPy
- **API Server:** Uvicorn

## Getting Started

### Requirements

- Python 3.11 or later
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/AAyesha-user/FaultTrace-AI.git
cd FaultTrace-AI
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

To start the FastAPI backend, open another terminal and run:

```bash
uvicorn api:app --reload
```

The API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

## Experiments

FaultTrace AI includes experiments to evaluate diagnostic methods and observe system behavior under different fault scenarios.

The available experiment scripts include:

```bash
python -m backend.experiments.run_experiments
python -m backend.experiments.run_multi_fault_experiments
python -m backend.experiments.run_stress_tests
```

Experiment results are saved in the `data/` directory.

## Research Inspiration

FaultTrace AI draws inspiration from the published research paper:

**"Diagnosis of Intermittent Faults in Multi-Agent Systems: An SFL Approach"**

Avraham Natan, Meir Kalech, and Roman Barták.  
*Artificial Intelligence, Volume 324, 2023, Article 103994.*

The paper provides a scientific foundation for exploring spectrum-based fault diagnosis in multi-agent systems. FaultTrace AI develops this direction into an interactive software prototype, with its own simulation workflow, diagnostic comparisons, experiments, dashboard, and recovery simulation.

The project is research-inspired and is not intended to be an exact reproduction of the original paper.

## Project Scope

FaultTrace AI is currently demonstrated in a simulated warehouse environment. Robots, fault scenarios, diagnostic results, and recovery outcomes are simulated in software. The prototype has not yet been integrated with physical robots or tested in a real industrial warehouse.

## Project

Developed as part of the **Capital Youth Expo (CYE) 2026 Innovation Sprint**.