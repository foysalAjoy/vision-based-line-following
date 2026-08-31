import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "worlds" / "line_following_curve_test.wbt"
WORLDS = ROOT / "worlds"

CONTROLLERS = ["p_line_follower", "pd_line_follower", "cnn_pd_hybrid_final"]

# Webots is deterministic: repeating one world reproduces the same
# trajectory exactly, so repeats give SD = 0.000 and are not replication.
# Spread has to come from varying the CONDITION, hence a sweep of start
# offsets. 0.000 is also the centred reference analyze_transient.py needs.
BASELINE_OFFSETS = [-0.045, -0.030, -0.015, 0.000, 0.015, 0.030, 0.045]

# Scene illumination multipliers. 1.00 is the unmodified world.
LUMINOSITY = [0.03, 0.05, 0.07, 0.10, 0.30, 1.00, 3.00, 8.00, 12.0, 16.0, 22.0, 30.0]
ROBUST_OFFSET = 0.030

if not BASE.exists():
    sys.exit(f"ERROR: {BASE} not found")

text = BASE.read_text(encoding="utf-8")

LIGHT_RE = re.compile(r"TexturedBackgroundLight\s*\{[^{}]*\}")

if not LIGHT_RE.search(text):
    sys.exit("ERROR: no TexturedBackgroundLight node found in the base "
             "world. The illumination sweep needs one.")
if 'name "view"' not in text:
    sys.exit("ERROR: base world has no Display named 'view'. "
             "Add it to turretSlot first.")
if "supervisor TRUE" not in text:
    sys.exit("ERROR: base world does not set supervisor TRUE on the E-puck.")

n_seg = text.count("baseColor 0.01 0.01 0.01")
if n_seg != 8:
    print(f"WARNING: expected 8 track segments, found {n_seg}. "
          f"Has the base world been edited?")

# Drop stale hidden wheel/joint state so every world starts clean.
text = "\n".join(l for l in text.splitlines()
                 if not l.strip().startswith("hidden "))


def build(controller, offset, luminosity=None):
    out = text
    if luminosity is not None:
        out = LIGHT_RE.sub(
            "TexturedBackgroundLight {\n"
            f"  luminosity {luminosity}\n"
            "}",
            out, count=1)
    out = re.sub(r'controller\s+"[^"]+"',
                 f'controller "{controller}"', out, count=1)
    args = f'controllerArgs [\n    "--offset={offset:.3f}"\n    "--quit"\n  ]'
    out = re.sub(r'controllerArgs\s*\[[^\]]*\]', args, out, count=1)
    return out


for old in WORLDS.glob("exp_*.wbt"):
    old.unlink()
for old in WORLDS.glob(".exp_*"):
    old.unlink()

written = []

for controller in CONTROLLERS:
    for offset in BASELINE_OFFSETS:
        mm = int(round(offset * 1000))
        sign = "n" if mm < 0 else "p"
        name = f"exp_baseline__{controller}_off{sign}{abs(mm):03d}.wbt"
        (WORLDS / name).write_text(build(controller, offset),
                                   encoding="utf-8")
        written.append(name)

for lum in LUMINOSITY:
    tag = f"{int(round(lum * 1000)):05d}"
    for controller in CONTROLLERS:
        name = f"exp_robust_c{tag}__{controller}.wbt"
        (WORLDS / name).write_text(
            build(controller, ROBUST_OFFSET, lum), encoding="utf-8")
        written.append(name)

print("=" * 62)
print(" EXPERIMENT WORLD GENERATOR")
print("=" * 62)
for name in written:
    print(f"  worlds/{name}")
print("=" * 62)
print(f" {len(written)} worlds written to {WORLDS}")
print(f"   baseline offsets : {[int(o*1000) for o in BASELINE_OFFSETS]} mm")
print(f"   luminosity levels: {LUMINOSITY}")
print(" Next: python scripts/run_all_experiments.py --repeats 1 --only baseline")
print("       python scripts/run_all_experiments.py --repeats 1 --only robust")
print("=" * 62)