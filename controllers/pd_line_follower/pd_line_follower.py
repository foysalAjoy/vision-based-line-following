import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from controller_core import run

run(method="PD", kp=1.80, kd=0.35, start_offset_y=0.030)