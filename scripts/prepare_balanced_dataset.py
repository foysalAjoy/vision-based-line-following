from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_NAMES = [
    "run_20260720_210122",
    "recovery_y_positive_005",
    "recovery_y_negative_005",
    "medium_y_positive_003",
    "medium_y_negative_003",
]

RANDOM_SEED = 42
CENTRE_SOURCE_LIMIT = 180
CENTRE_THRESHOLD = 0.05
STRONG_THRESHOLD = 0.15


def steering_category(target: float) -> str:
    if target <= -STRONG_THRESHOLD:
        return "strong_negative"
    if target < -CENTRE_THRESHOLD:
        return "medium_negative"
    if target <= CENTRE_THRESHOLD:
        return "centre"
    if target < STRONG_THRESHOLD:
        return "medium_positive"
    return "strong_positive"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_directory = (
        project_root
        / "data"
        / "cnn_dataset"
        / "runs"
    )

    output_directory = (
        project_root
        / "data"
        / "cnn_dataset"
        / "prepared_v1"
    )

    images_output = output_directory / "images"

    if output_directory.exists():
        shutil.rmtree(output_directory)

    images_output.mkdir(parents=True, exist_ok=True)

    raw_records: list[dict[str, object]] = []

    for run_name in RUN_NAMES:
        run_directory = runs_directory / run_name
        labels_path = run_directory / "labels.csv"
        images_directory = run_directory / "images"

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Missing labels file: {labels_path}"
            )

        labels = pd.read_csv(labels_path)

        if "steering_target" not in labels.columns:
            raise ValueError(
                f"{run_name} has no steering_target column."
            )

        image_count = len(
            list(images_directory.glob("*.png"))
        )

        if image_count != len(labels):
            raise ValueError(
                f"{run_name}: {image_count} images but "
                f"{len(labels)} labels."
            )

        for _, row in labels.iterrows():
            image_filename = str(row["image_filename"])
            image_path = images_directory / image_filename

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing image: {image_path}"
                )

            target = float(row["steering_target"])
            sample_id = int(row["sample_id"])

            raw_records.append(
                {
                    "source_run": run_name,
                    "source_sample_id": sample_id,
                    "source_image_filename": image_filename,
                    "source_image_path": str(image_path),
                    "steering_target": target,
                    "category": steering_category(target),
                }
            )

    raw_data = pd.DataFrame(raw_records)

    centre_data = raw_data[
        raw_data["steering_target"].abs()
        <= CENTRE_THRESHOLD
    ]

    non_centre_data = raw_data[
        raw_data["steering_target"].abs()
        > CENTRE_THRESHOLD
    ]

    centre_sample_count = min(
        CENTRE_SOURCE_LIMIT,
        len(centre_data),
    )

    selected_centre = centre_data.sample(
        n=centre_sample_count,
        random_state=RANDOM_SEED,
    )

    selected_sources = pd.concat(
        [non_centre_data, selected_centre],
        ignore_index=True,
    )

    selected_sources = selected_sources.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    prepared_records: list[dict[str, object]] = []

    output_index = 0

    for _, row in selected_sources.iterrows():
        source_path = Path(
            str(row["source_image_path"])
        )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV could not read: {source_path}"
            )

        source_run = str(row["source_run"])
        source_sample_id = int(
            row["source_sample_id"]
        )
        original_target = float(
            row["steering_target"]
        )

        group_id = (
            f"{source_run}__{source_sample_id:06d}"
        )

        # Save original image.
        output_index += 1
        original_filename = (
            f"sample_{output_index:06d}_original.png"
        )

        original_output_path = (
            images_output / original_filename
        )

        if not cv2.imwrite(
            str(original_output_path),
            image,
        ):
            raise RuntimeError(
                f"Failed to save {original_output_path}"
            )

        prepared_records.append(
            {
                "prepared_sample_id": output_index,
                "group_id": group_id,
                "image_filename": original_filename,
                "steering_target": original_target,
                "category": steering_category(
                    original_target
                ),
                "augmentation": "original",
                "source_run": source_run,
                "source_sample_id": source_sample_id,
                "source_image_filename": str(
                    row["source_image_filename"]
                ),
            }
        )

        # Horizontal reflection: reverse target direction.
        mirrored_image = cv2.flip(image, 1)
        mirrored_target = float(
            np.clip(-original_target, -1.0, 1.0)
        )

        output_index += 1
        mirrored_filename = (
            f"sample_{output_index:06d}_mirrored.png"
        )

        mirrored_output_path = (
            images_output / mirrored_filename
        )

        if not cv2.imwrite(
            str(mirrored_output_path),
            mirrored_image,
        ):
            raise RuntimeError(
                f"Failed to save {mirrored_output_path}"
            )

        prepared_records.append(
            {
                "prepared_sample_id": output_index,
                "group_id": group_id,
                "image_filename": mirrored_filename,
                "steering_target": mirrored_target,
                "category": steering_category(
                    mirrored_target
                ),
                "augmentation": "horizontal_reflection",
                "source_run": source_run,
                "source_sample_id": source_sample_id,
                "source_image_filename": str(
                    row["source_image_filename"]
                ),
            }
        )

    prepared_data = pd.DataFrame(
        prepared_records
    )

    manifest_path = (
        output_directory / "manifest.csv"
    )
    prepared_data.to_csv(
        manifest_path,
        index=False,
    )

    raw_category_counts = (
        raw_data["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="samples")
    )

    raw_category_counts.to_csv(
        output_directory
        / "raw_category_counts.csv",
        index=False,
    )

    prepared_category_counts = (
        prepared_data["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="samples")
    )

    prepared_category_counts.to_csv(
        output_directory
        / "prepared_category_counts.csv",
        index=False,
    )

    prepared_targets = prepared_data[
        "steering_target"
    ].astype(float)

    summary = pd.DataFrame(
        [
            {
                "raw_samples": len(raw_data),
                "selected_source_samples": (
                    len(selected_sources)
                ),
                "prepared_samples": len(
                    prepared_data
                ),
                "selected_centre_sources": (
                    centre_sample_count
                ),
                "selected_non_centre_sources": (
                    len(non_centre_data)
                ),
                "target_minimum": float(
                    prepared_targets.min()
                ),
                "target_maximum": float(
                    prepared_targets.max()
                ),
                "target_mean": float(
                    prepared_targets.mean()
                ),
                "target_standard_deviation": float(
                    prepared_targets.std()
                ),
                "centre_percentage": float(
                    (
                        prepared_targets.abs()
                        <= CENTRE_THRESHOLD
                    ).mean()
                    * 100
                ),
                "medium_percentage": float(
                    (
                        (
                            prepared_targets.abs()
                            > CENTRE_THRESHOLD
                        )
                        & (
                            prepared_targets.abs()
                            < STRONG_THRESHOLD
                        )
                    ).mean()
                    * 100
                ),
                "strong_percentage": float(
                    (
                        prepared_targets.abs()
                        >= STRONG_THRESHOLD
                    ).mean()
                    * 100
                ),
            }
        ]
    )

    summary_path = (
        output_directory
        / "prepared_dataset_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.hist(
        prepared_targets,
        bins=np.linspace(-1.0, 1.0, 41),
        edgecolor="black",
    )
    plt.xlabel("Steering target")
    plt.ylabel("Number of samples")
    plt.title(
        "Balanced Prototype Dataset Steering Distribution"
    )
    plt.tight_layout()

    histogram_path = (
        output_directory
        / "prepared_target_histogram.png"
    )
    plt.savefig(histogram_path, dpi=200)
    plt.close()

    print("=" * 76)
    print("BALANCED PROTOTYPE DATASET PREPARATION")
    print("=" * 76)
    print(f"Raw samples              : {len(raw_data)}")
    print(
        f"Selected centre sources  : "
        f"{centre_sample_count}"
    )
    print(
        f"Selected non-centre      : "
        f"{len(non_centre_data)}"
    )
    print(
        f"Selected source samples  : "
        f"{len(selected_sources)}"
    )
    print(
        f"Prepared samples         : "
        f"{len(prepared_data)}"
    )
    print(
        f"Prepared target range    : "
        f"{prepared_targets.min():+.4f} to "
        f"{prepared_targets.max():+.4f}"
    )
    print(
        f"Prepared target mean     : "
        f"{prepared_targets.mean():+.6f}"
    )
    print(
        f"Centre percentage        : "
        f"{(
            (
                prepared_targets.abs()
                <= CENTRE_THRESHOLD
            ).mean() * 100
        ):.2f}%"
    )
    print(
        f"Medium percentage        : "
        f"{(
            (
                (
                    prepared_targets.abs()
                    > CENTRE_THRESHOLD
                )
                & (
                    prepared_targets.abs()
                    < STRONG_THRESHOLD
                )
            ).mean() * 100
        ):.2f}%"
    )
    print(
        f"Strong percentage        : "
        f"{(
            (
                prepared_targets.abs()
                >= STRONG_THRESHOLD
            ).mean() * 100
        ):.2f}%"
    )

    print()
    print("Prepared category counts:")
    print(
        prepared_category_counts.to_string(
            index=False
        )
    )

    print()
    print(f"Manifest saved : {manifest_path}")
    print(f"Summary saved  : {summary_path}")
    print(f"Histogram saved: {histogram_path}")


if __name__ == "__main__":
    main()
