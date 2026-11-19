
import random

from backend.simulation.models import Robot, Task, ExecutionRecord


class WarehouseSimulation:
    def __init__(
        self,
        num_robots=5,
        seed=42,
        faulty_robot_ids=None,
        fault_probability=0.30,
    ):
        self.random = random.Random(seed)

        self.robots = [
            Robot(robot_id=f"R{i + 1}", fault_probability=0.0)
            for i in range(num_robots)
        ]

        faulty_robot_ids = faulty_robot_ids or []

        for robot in self.robots:
            if robot.robot_id in faulty_robot_ids:
                robot.fault_probability = fault_probability

        self.tasks = []

    def generate_tasks(self, num_tasks=50):
        task_types = ["transport", "inspection", "picking"]

        for i in range(num_tasks):
            task = Task(
                task_id=f"T{i + 1}",
                task_type=self.random.choice(task_types),
                difficulty=self.random.uniform(0.5, 1.5),
            )

            self.tasks.append(task)

        return self.tasks

    def run(self):
        records = []

        if not self.tasks:
            self.generate_tasks()

        for task in self.tasks:
            robot = self.random.choice(self.robots)

            robot.total_tasks += 1

            faulty_behavior = (
                self.random.random() < robot.fault_probability
            )

            normal_failure_probability = 0.05 * task.difficulty
            failure_probability = normal_failure_probability

            if faulty_behavior:
                failure_probability += 0.70

            failure_probability = min(failure_probability, 1.0)

            success = (
                self.random.random() >= failure_probability
            )

            if success:
                robot.successful_tasks += 1
            else:
                robot.failed_tasks += 1

            record = ExecutionRecord(
                task_id=task.task_id,
                robot_id=robot.robot_id,
                task_type=task.task_type,
                success=success,
                faulty_behavior=faulty_behavior,
                difficulty=task.difficulty,
            )

            records.append(record)

        return records
