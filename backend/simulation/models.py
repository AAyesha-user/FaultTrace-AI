from dataclasses import dataclass, asdict


@dataclass
class Robot:
    robot_id: str
    fault_probability: float = 0.0
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0


@dataclass
class Task:
    task_id: str
    task_type: str
    difficulty: float = 1.0


@dataclass
class ExecutionRecord:
    task_id: str
    robot_id: str
    task_type: str
    success: bool
    faulty_behavior: bool
    difficulty: float

    def to_dict(self):
        return asdict(self)