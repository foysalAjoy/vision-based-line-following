from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = [
    ("numpy", "NumPy"),
    ("cv2", "OpenCV"),
    ("matplotlib", "Matplotlib"),
    ("pandas", "Pandas"),
    ("tensorflow", "TensorFlow"),
    ("PIL", "Pillow"),
]

PATHS = [
    ("Trained CNN", ROOT / "models_trained" / "line_error_cnn.keras"),
    ("Training history", ROOT / "models_trained" / "training_history.csv"),
    ("Prepared manifest",
     ROOT / "data" / "cnn_dataset" / "prepared_v1" / "manifest_line_error.csv"),
    ("Base world", ROOT / "worlds" / "line_following_curve_test.wbt"),
]


def main() -> int:
    print("=" * 65)
    print("VISION-BASED LINE FOLLOWING ENVIRONMENT CHECK")
    print("=" * 65)
    print(f"Operating system : {platform.platform()}")
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print()

    missing = []
    print("Packages:")
    for module, label in PACKAGES:
        try:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "unknown")
            print(f"  {label:<16} {version}")
        except Exception as exc:
            missing.append(label)
            print(f"  {label:<16} MISSING  ({type(exc).__name__})")

    print()
    print("Project files:")
    absent = []
    for label, path in PATHS:
        ok = path.exists()
        if not ok:
            absent.append(label)
        print(f"  {label:<18} {'OK' if ok else 'MISSING'}  "
              f"{path.relative_to(ROOT)}")

    print()
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"TensorFlow GPUs : {len(gpus)}"
              + (f"  ({gpus[0].name})" if gpus else "  (running on CPU)"))
    except Exception:
        print("TensorFlow GPUs : could not query")

    print()
    if missing:
        print("RESULT: FAILED - install the missing packages:")
        print("  pip install -r requirements.txt")
        return 1
    if absent:
        print("RESULT: PARTIAL - packages fine, but these are missing:")
        for a in absent:
            print(f"  - {a}")
        return 0

    print("ENVIRONMENT CHECK PASSED")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
