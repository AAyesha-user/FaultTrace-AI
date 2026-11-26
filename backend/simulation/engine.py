
import random

from backend.simulation.models import Robot, Task, ExecutionRecord


class WarehouseSimulation:

    def __init__(
        self,
        num_robots=5,
        seed=42,
        faulty_robot_ids=None,
        fault_probability=0.30,
        fault_mode="Intermittent",
        fault_severity=0.70,
    ):

        self.random = random.Random(seed)

        self.num_robots = num_robots

        if num_robots < 1:
            raise ValueError("At least one robot is required.")

        if not 0 <= fault_probability <= 1:
            raise ValueError("Fault probability must be between 0 and 1.")

        if not 0 <= fault_severity <= 1:
            raise ValueError("Fault severity must be between 0 and 1.")

        self.fault_mode = fault_mode.strip().lower()

        if self.fault_mode not in ["intermittent", "permanent"]:
            raise ValueError(
                "Fault mode must be Intermittent or Permanent."
            )

        self.fault_probability = fault_probability
        self.fault_severity = fault_severity

        self.faulty_robot_ids = faulty_robot_ids or []

        self.robots = [
            Robot(
                robot_id=f"R{i + 1}",
                fault_probability=0.0
            )
            for i in range(num_robots)
        ]

        for robot in self.robots:

            if robot.robot_id in self.faulty_robot_ids:

                if self.fault_mode == "Permanent":
                    robot.fault_probability = 1.0

                else:
                    robot.fault_probability = fault_probability

        self.tasks = []

    def generate_tasks(self, num_tasks=50):

        task_types = [
            "transport",
            "inspection",
            "picking"
        ]

        self.tasks = []

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

            # Determine whether the robot exhibits faulty behaviour.

            if robot.robot_id in self.faulty_robot_ids:

                if self.fault_mode == "Permanent":
                    faulty_behavior = True

                else:
                    faulty_behavior = (
                        self.random.random()
                        < self.fault_probability
                    )

            else:
                faulty_behavior = False

            # Calculate normal operational failure probability.

            normal_failure_probability = (
                0.05 * task.difficulty
            )

            failure_probability = normal_failure_probability

            # Apply judge-controlled fault severity.

            if faulty_behavior:

                failure_probability += self.fault_severity

            failure_probability = min(
                max(failure_probability, 0.0),
                1.0
            )

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
