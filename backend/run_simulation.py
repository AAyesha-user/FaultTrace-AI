import csv
from pathlib import Path

from backend.simulation.engine import WarehouseSimulation


def main():
    simulation = WarehouseSimulation(num_robots=5, seed=42)
    simulation.generate_tasks(num_tasks=100)

    records = simulation.run()

    # Create the data folder automatically.
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Export task-level execution history.
    output_file = data_dir / "execution_history.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].to_dict().keys()
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record.to_dict())

    print("\n=== FaultTrace AI: Simulation Report ===")
    print(f"Tasks simulated: {len(records)}")
    print(f"Execution history saved to: {output_file}")

    print("\nRobot performance:")

    for robot in simulation.robots:
        print(
            f"{robot.robot_id}: "
            f"{robot.total_tasks} tasks | "
            f"{robot.successful_tasks} successful | "
            f"{robot.failed_tasks} failed"
        )


if __name__ == "__main__":
    main()