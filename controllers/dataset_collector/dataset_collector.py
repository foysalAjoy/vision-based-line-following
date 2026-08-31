from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from controller import Robot


# Controller settings
KP = 1.8
KD = 0.25

BASE_SPEED = 1.2
MAX_SPEED = 3.0
MAX_CORRECTION = 1.5

DERIVATIVE_ALPHA = 0.25
RUN_DURATION_SECONDS = 30.0
SAVE_EVERY_N_STEPS = 2

ROI_HEIGHT_RATIO = 0.72
OUTPUT_WIDTH = 160
OUTPUT_HEIGHT = 96

MIN_CONTOUR_AREA = 30.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def main() -> None:
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())
    time_step_seconds = time_step / 1000.0

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    camera = robot.getDevice("downward camera")

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    camera.enable(time_step)

    project_root = Path(__file__).resolve().parents[2]

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_directory = (
        project_root
        / "data"
        / "cnn_dataset"
        / "runs"
        / run_id
    )

    image_directory = run_directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)

    labels_path = run_directory / "labels.csv"

    fieldnames = [
        "sample_id",
        "image_filename",
        "simulation_time",
        "normalized_error",
        "steering_correction",
        "steering_target",
        "left_speed",
        "right_speed",
        "line_detected",
        "contour_area",
    ]

    previous_error = 0.0
    filtered_derivative = 0.0
    sample_count = 0
    simulation_step = 0

    csv_file = labels_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    )
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    print("=" * 72)
    print("CNN DATASET COLLECTION STARTED")
    print("=" * 72)
    print(f"Run ID             : {run_id}")
    print(f"Duration           : {RUN_DURATION_SECONDS:.1f} seconds")
    print(f"Save interval      : every {SAVE_EVERY_N_STEPS} steps")
    print(f"Image resolution   : {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")
    print(f"Kp / Kd            : {KP} / {KD}")
    print(f"Dataset directory  : {run_directory}")

    while robot.step(time_step) != -1:
        current_time = robot.getTime()

        if current_time >= RUN_DURATION_SECONDS:
            break

        simulation_step += 1

        image_bytes = camera.getImage()

        if image_bytes is None:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            continue

        width = camera.getWidth()
        height = camera.getHeight()

        bgra_frame = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        ).reshape((height, width, 4))

        bgr_frame = bgra_frame[:, :, :3]

        roi_height = int(height * ROI_HEIGHT_RATIO)
        roi = bgr_frame[:roi_height, :]

        grayscale = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            grayscale,
            (5, 5),
            0,
        )

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = np.ones((3, 3), dtype=np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        line_detected = False
        contour_area = 0.0
        normalized_error = previous_error

        if contours:
            largest_contour = max(
                contours,
                key=cv2.contourArea,
            )

            contour_area = float(
                cv2.contourArea(largest_contour)
            )

            moments = cv2.moments(largest_contour)

            if (
                contour_area >= MIN_CONTOUR_AREA
                and moments["m00"] != 0
            ):
                centroid_x = int(
                    moments["m10"] / moments["m00"]
                )

                image_centre_x = width // 2

                normalized_error = (
                    centroid_x - image_centre_x
                ) / image_centre_x

                normalized_error = clamp(
                    normalized_error,
                    -1.0,
                    1.0,
                )

                line_detected = True

        if line_detected:
            raw_derivative = (
                normalized_error - previous_error
            ) / time_step_seconds

            filtered_derivative = (
                DERIVATIVE_ALPHA * raw_derivative
                + (1.0 - DERIVATIVE_ALPHA)
                * filtered_derivative
            )

            proportional_term = KP * normalized_error
            derivative_term = KD * filtered_derivative

            correction = clamp(
                proportional_term + derivative_term,
                -MAX_CORRECTION,
                MAX_CORRECTION,
            )

            left_speed = clamp(
                BASE_SPEED + correction,
                -MAX_SPEED,
                MAX_SPEED,
            )

            right_speed = clamp(
                BASE_SPEED - correction,
                -MAX_SPEED,
                MAX_SPEED,
            )

            previous_error = normalized_error

        else:
            correction = 0.0
            left_speed = 0.0
            right_speed = 0.0

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        should_save = (
            line_detected
            and simulation_step % SAVE_EVERY_N_STEPS == 0
        )

        if should_save:
            sample_count += 1

            image_filename = (
                f"frame_{sample_count:06d}.png"
            )

            processed_image = cv2.resize(
                grayscale,
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            image_path = image_directory / image_filename

            saved_successfully = cv2.imwrite(
                str(image_path),
                processed_image,
            )

            if not saved_successfully:
                raise RuntimeError(
                    f"Failed to save image: {image_path}"
                )

            steering_target = clamp(
                correction / MAX_CORRECTION,
                -1.0,
                1.0,
            )

            writer.writerow(
                {
                    "sample_id": sample_count,
                    "image_filename": image_filename,
                    "simulation_time": (
                        f"{current_time:.4f}"
                    ),
                    "normalized_error": (
                        f"{normalized_error:.6f}"
                    ),
                    "steering_correction": (
                        f"{correction:.6f}"
                    ),
                    "steering_target": (
                        f"{steering_target:.6f}"
                    ),
                    "left_speed": (
                        f"{left_speed:.6f}"
                    ),
                    "right_speed": (
                        f"{right_speed:.6f}"
                    ),
                    "line_detected": 1,
                    "contour_area": (
                        f"{contour_area:.2f}"
                    ),
                }
            )

            if sample_count % 50 == 0:
                print(
                    f"Saved {sample_count:4d} samples | "
                    f"error={normalized_error:+.3f} | "
                    f"target={steering_target:+.3f}"
                )

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    csv_file.close()

    print("=" * 72)
    print("CNN DATASET COLLECTION COMPLETED")
    print("=" * 72)
    print(f"Images saved : {sample_count}")
    print(f"Labels saved : {labels_path}")
    print(f"Run folder   : {run_directory}")


if __name__ == "__main__":
    main()
