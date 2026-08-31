import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "worlds" / "line_following_curve_test.wbt"
OUTDIR = ROOT / "worlds" / "robustness"

OLD_COLOR = "baseColor 0.01 0.01 0.01"
CONTRASTS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]

if not BASE.exists():
    sys.exit(f"ERROR: {BASE} not found")

text = BASE.read_text(encoding="utf-8")

n_seg = text.count(OLD_COLOR)
if n_seg != 8:
    sys.exit(f"ERROR: expected 8 track segments, found {n_seg}.\n"
             f"Has the base world been edited? Check '{OLD_COLOR}'.")

if 'name "view"' not in text:
    sys.exit("ERROR: base world has no Display named 'view'. "
             "Add it to turretSlot first.")
if "supervisor TRUE" not in text:
    sys.exit("ERROR: base world does not set supervisor TRUE on the E-puck.")

# Drop stale hidden wheel/joint state so every world starts clean.
text = "\n".join(l for l in text.splitlines()
                 if not l.strip().startswith("hidden "))

OUTDIR.mkdir(parents=True, exist_ok=True)
for old in OUTDIR.glob("robust_c*.wbt"):
    old.unlink()

print("=" * 56)
print(" ROBUSTNESS WORLD GENERATOR")
print("=" * 56)

for c in CONTRASTS:
    out = text.replace(OLD_COLOR, f"baseColor {c} {c} {c}")
    out = re.sub(r'controller\s+"[^"]+"',
                 'controller "p_line_follower"', out, count=1)

              # 0.015 -> 0015, no collision
    tag = f"{int(round(c * 1000)):04d}"
    path = OUTDIR / f"robust_c{tag}.wbt"
    path.write_text(out, encoding="utf-8")
    print(f"  baseColor {c:<5}  ->  worlds/robustness/{path.name}")

print("=" * 56)
print(f" {len(CONTRASTS)} worlds written to {OUTDIR}")
print("")
print(" Each world starts with controller \"p_line_follower\".")
print(" Change the controller field per run, exactly as before.")
print("=" * 56)