from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_NAMES = [
    "run_20260720_210122",
    "recovery_y_positive_005",
    "recovery_y_negative_005",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_directory = project_root / "data" / "cnn_dataset" / "runs"

    output_directory = (
        project_root
        / "data"
        / "cnn_dataset"
        / "combined_audit"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    combined_frames: list[pd.DataFrame] = []
    run_summaries: list[dict[str, object]] = []

    for run_name in RUN_NAMES:
        run_directory = runs_directory / run_name
        labels_path = run_directory / "labels.csv"
        images_directory = run_directory / "images"

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Missing labels file: {labels_path}"
            )

        data = pd.read_csv(labels_path)

        image_count = len(
            list(images_directory.glob("*.png"))
        )

        if image_count != len(data):
            raise ValueError(
                f"{run_name}: {image_count} images but "
                f"{len(data)} label rows"
            )

        data["source_run"] = run_name

        data["source_image_path"] = data[
            "image_filename"
        ].apply(
            lambda filename: str(
                images_directory / str(filename)
            )
        )

        target = pd.to_numeric(
            data["steering_target"],
            errors="coerce",
        )

        if target.isna().any():
            raise ValueError(
                f"{run_name} contains invalid targets."
            )

        run_summaries.append(
            {
                "run_name": run_name,
                "samples": len(data),
                "target_minimum": float(target.min()),
                "target_maximum": float(target.max()),
                "target_mean": float(target.mean()),
                "target_standard_deviation": float(
                    target.std()
                ),
                "negative_strong": int(
                    (target <= -0.15).sum()
                ),
                "positive_strong": int(
                    (target >= 0.15).sum()
                ),
                "saturated_negative": int(
                    (target <= -0.999).sum()
                ),
                "saturated_positive": int(
                    (target >= 0.999).sum()
                ),
            }
        )

        combined_frames.append(data)

    combined = pd.concat(
        combined_frames,
        ignore_index=True,
    )

    target = pd.to_numeric(
        combined["steering_target"],
        errors="coerce",
    )

    absolute_target = target.abs()

    negative_count = int((target < -0.05).sum())
    centre_count = int(
        ((target >= -0.05) & (target <= 0.05)).sum()
    )
    positive_count = int((target > 0.05).sum())

    negative_strong = int((target <= -0.15).sum())
    positive_strong = int((target >= 0.15).sum())
    saturated_count = int(
        (absolute_target >= 0.999).sum()
    )

    combined_summary = pd.DataFrame(
        [
            {
                "total_samples": len(combined),
                "target_minimum": float(target.min()),
                "target_maximum": float(target.max()),
                "target_mean": float(target.mean()),
                "target_standard_deviation": float(
                    target.std()
                ),
                "negative_samples_below_minus_0.05": (
                    negative_count
                ),
                "centre_samples_within_0.05": centre_count,
                "positive_samples_above_0.05": (
                    positive_count
                ),
                "negative_strong_samples": negative_strong,
                "positive_strong_samples": positive_strong,
                "saturated_samples": saturated_count,
                "percentage_within_0.05": float(
                    (absolute_target <= 0.05).mean()
                    * 100
                ),
                "percentage_strong": float(
                    (absolute_target >= 0.15).mean()
                    * 100
                ),
            }
        ]
    )

    manifest_path = (
        output_directory
        / "combined_dataset_manifest.csv"
    )
    combined.to_csv(manifest_path, index=False)

    run_summary_path = (
        output_directory
        / "run_comparison.csv"
    )
    pd.DataFrame(run_summaries).to_csv(
        run_summary_path,
        index=False,
    )

    combined_summary_path = (
        output_directory
        / "combined_dataset_summary.csv"
    )
    combined_summary.to_csv(
        combined_summary_path,
        index=False,
    )

    print("=" * 76)
    print("COMBINED CNN DATASET AUDIT")
    print("=" * 76)

    print(pd.DataFrame(run_summaries).to_string(index=False))

    print()
    print("-" * 76)
    print(f"Total samples          : {len(combined)}")
    print(
        f"Target range           : "
        f"{target.min():+.4f} to {target.max():+.4f}"
    )
    print(
        f"Target mean            : {target.mean():+.4f}"
    )
    print(
        f"Target standard dev.   : {target.std():.4f}"
    )
    print(f"Negative samples       : {negative_count}")
    print(f"Centre samples         : {centre_count}")
    print(f"Positive samples       : {positive_count}")
    print(f"Strong negative        : {negative_strong}")
    print(f"Strong positive        : {positive_strong}")
    print(f"Saturated samples      : {saturated_count}")
    print(
        f"|target| <= 0.05       : "
        f"{(absolute_target <= 0.05).mean() * 100:.2f}%"
    )
    print(
        f"|target| >= 0.15       : "
        f"{(absolute_target >= 0.15).mean() * 100:.2f}%"
    )

    plt.figure(figsize=(10, 5))
    plt.hist(
        target,
        bins=np.linspace(-1.0, 1.0, 41),
        edgecolor="black",
    )
    plt.xlabel("Steering target")
    plt.ylabel("Number of samples")
    plt.title("Combined CNN Dataset Steering Distribution")
    plt.tight_layout()

    histogram_path = (
        output_directory
        / "combined_target_histogram.png"
    )
    plt.savefig(histogram_path, dpi=200)
    plt.close()

    print()
    print(f"Manifest saved : {manifest_path}")
    print(f"Summary saved  : {combined_summary_path}")
    print(f"Histogram saved: {histogram_path}")


if __name__ == "__main__":
    main()
