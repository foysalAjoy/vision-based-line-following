from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_directory = project_root / "data" / "cnn_dataset" / "runs"

    run_directories = [
        path
        for path in runs_directory.iterdir()
        if path.is_dir()
    ]

    if not run_directories:
        raise FileNotFoundError(
            f"No dataset runs found in {runs_directory}"
        )

    latest_run = max(
        run_directories,
        key=lambda path: path.stat().st_mtime,
    )

    labels_path = latest_run / "labels.csv"
    images_directory = latest_run / "images"
    analysis_directory = latest_run / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}"
        )

    data = pd.read_csv(labels_path)

    required_columns = {
        "sample_id",
        "image_filename",
        "normalized_error",
        "steering_correction",
        "steering_target",
        "left_speed",
        "right_speed",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    image_files = sorted(images_directory.glob("*.png"))
    expected_images = {
        str(filename)
        for filename in data["image_filename"]
    }
    existing_images = {
        image_path.name
        for image_path in image_files
    }

    missing_images = sorted(
        expected_images.difference(existing_images)
    )
    unlisted_images = sorted(
        existing_images.difference(expected_images)
    )

    steering_target = pd.to_numeric(
        data["steering_target"],
        errors="coerce",
    )
    normalized_error = pd.to_numeric(
        data["normalized_error"],
        errors="coerce",
    )

    invalid_target_count = int(
        steering_target.isna().sum()
    )
    outside_range_count = int(
        ((steering_target < -1.0) |
         (steering_target > 1.0)).sum()
    )

    target_absolute = steering_target.abs()

    summary = {
        "run_name": latest_run.name,
        "label_rows": len(data),
        "image_files": len(image_files),
        "missing_images": len(missing_images),
        "unlisted_images": len(unlisted_images),
        "invalid_target_rows": invalid_target_count,
        "targets_outside_range": outside_range_count,
        "target_minimum": float(steering_target.min()),
        "target_maximum": float(steering_target.max()),
        "target_mean": float(steering_target.mean()),
        "target_standard_deviation": float(
            steering_target.std()
        ),
        "mean_absolute_target": float(
            target_absolute.mean()
        ),
        "percentage_target_below_0.05": float(
            (target_absolute < 0.05).mean() * 100
        ),
        "percentage_target_0.05_to_0.15": float(
            (
                (target_absolute >= 0.05)
                & (target_absolute < 0.15)
            ).mean() * 100
        ),
        "percentage_target_at_least_0.15": float(
            (target_absolute >= 0.15).mean() * 100
        ),
        "error_minimum": float(normalized_error.min()),
        "error_maximum": float(normalized_error.max()),
    }

    summary_path = analysis_directory / "dataset_summary.csv"
    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False,
    )

    print("=" * 72)
    print("CNN PILOT DATASET AUDIT")
    print("=" * 72)
    print(f"Run                     : {latest_run.name}")
    print(f"Label rows              : {len(data)}")
    print(f"Image files             : {len(image_files)}")
    print(f"Missing images          : {len(missing_images)}")
    print(f"Unlisted images         : {len(unlisted_images)}")
    print(f"Invalid target rows     : {invalid_target_count}")
    print(f"Targets outside [-1,1]  : {outside_range_count}")
    print()
    print(
        f"Target range            : "
        f"{steering_target.min():+.4f} to "
        f"{steering_target.max():+.4f}"
    )
    print(
        f"Target mean             : "
        f"{steering_target.mean():+.4f}"
    )
    print(
        f"Target standard dev.    : "
        f"{steering_target.std():.4f}"
    )
    print(
        f"Mean absolute target    : "
        f"{target_absolute.mean():.4f}"
    )
    print(
        f"|target| < 0.05         : "
        f"{(target_absolute < 0.05).mean() * 100:.2f}%"
    )
    print(
        f"0.05 <= |target| < 0.15 : "
        f"{(
            (
                (target_absolute >= 0.05)
                & (target_absolute < 0.15)
            ).mean() * 100
        ):.2f}%"
    )
    print(
        f"|target| >= 0.15        : "
        f"{(target_absolute >= 0.15).mean() * 100:.2f}%"
    )

    plt.figure(figsize=(9, 5))
    plt.hist(
        steering_target.dropna(),
        bins=np.linspace(-1.0, 1.0, 41),
        edgecolor="black",
    )
    plt.xlabel("Steering target")
    plt.ylabel("Number of samples")
    plt.title("Pilot Dataset Steering-Target Distribution")
    plt.tight_layout()

    histogram_path = (
        analysis_directory
        / "steering_target_histogram.png"
    )
    plt.savefig(histogram_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        data["sample_id"],
        steering_target,
    )
    plt.axhline(0, linewidth=1)
    plt.xlabel("Sample ID")
    plt.ylabel("Steering target")
    plt.title("Steering Targets Across Pilot Collection Run")
    plt.tight_layout()

    sequence_path = (
        analysis_directory
        / "steering_target_sequence.png"
    )
    plt.savefig(sequence_path, dpi=200)
    plt.close()

    sample_count = min(12, len(data))
    selected_indices = np.linspace(
        0,
        len(data) - 1,
        sample_count,
        dtype=int,
    )

    contact_images = []

    for index in selected_indices:
        row = data.iloc[index]
        image_path = (
            images_directory
            / str(row["image_filename"])
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

        label = (
            f"ID {int(row['sample_id'])}  "
            f"T {float(row['steering_target']):+.3f}"
        )

        cv2.putText(
            image,
            label,
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        contact_images.append(image)

    if contact_images:
        while len(contact_images) < 12:
            contact_images.append(
                np.zeros_like(contact_images[0])
            )

        rows = []

        for start_index in range(0, 12, 4):
            row_image = cv2.hconcat(
                contact_images[
                    start_index:start_index + 4
                ]
            )
            rows.append(row_image)

        contact_sheet = cv2.vconcat(rows)

        contact_sheet_path = (
            analysis_directory
            / "sample_contact_sheet.png"
        )

        cv2.imwrite(
            str(contact_sheet_path),
            contact_sheet,
        )

        print(f"Contact sheet saved     : {contact_sheet_path}")

    print()
    print(f"Summary saved           : {summary_path}")
    print(f"Histogram saved         : {histogram_path}")
    print(f"Target sequence saved   : {sequence_path}")


if __name__ == "__main__":
    main()
