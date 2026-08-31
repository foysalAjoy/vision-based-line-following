import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from controller_core import run

run(
    method="CNN+PD",
    kp=1.80,
    kd=0.35,
    start_offset_y=0.030,
    model_path=ROOT / "models_trained" / "line_error_cnn.keras",
)