from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation_results_final"
OUT = ROOT / "evaluation_summary_final"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {"p": "P", "pd": "PD", "cnnpd": "CNN+PD Hybrid"}
ORDER = ["P", "PD", "CNN+PD Hybrid"]

SAT_ERROR = 0.95        # |perceived_error| above this counts as saturated
MAX_CORRECTION = 2.20   # must match common/controller_core.py
CLAMP_EPS = 0.01
RECOVER_BAND_MM = 3.0


def parse(stem):
    head, _, run = stem.rpartition("_run_")
    if not head:
        return None, None, 0
    tag, _, world = head.partition("__")
    try:
        return tag, world, int(run)
    except ValueError:
        return tag, world, 0


rows = []
traces = {}

for f in sorted(RES.glob("*_run_*.csv")):
    tag, world, run_no = parse(f.stem)
    if tag is None or world != "baseline":
        continue

    d = pd.read_csv(f)
    if d.empty or "cte_m" not in d.columns:
        continue
    num = d.apply(pd.to_numeric, errors="coerce")
    if num["cte_m"].isna().all():
        continue

    t = num["time_s"].to_numpy(float)
    cte = num["cte_m"].to_numpy(float)
    err = num["perceived_error"].to_numpy(float)
    corr = num["correction"].to_numpy(float)
    offset = float(num["start_offset_mm"].iloc[0]) if "start_offset_mm" in num else 0.0

    sat = float(np.mean(np.abs(err) >= SAT_ERROR)) * 100
    clamped = float(np.mean(np.abs(np.abs(corr) - MAX_CORRECTION)
                            < CLAMP_EPS)) * 100

    # how long until |cte| stays inside the band for the rest of the run
    band = RECOVER_BAND_MM / 1000.0
    recover = float("nan")
    for i in range(cte.size):
        if np.all(np.abs(cte[i:]) <= band):
            recover = float(t[i])
            break

    method = LABEL.get(tag, tag)
    rows.append({
        "Method": method,
        "Offset_mm": round(offset, 1),
        "Run": run_no,
        "Saturated_%": round(sat, 2),
        "Clamped_%": round(clamped, 2),
        "err_mean_abs": round(float(np.abs(err).mean()), 4),
        "err_std": round(float(err.std()), 4),
        "err_min": round(float(err.min()), 3),
        "err_max": round(float(err.max()), 3),
        "corr_mean_abs": round(float(np.abs(corr).mean()), 4),
        "Recover3mm_s": round(recover, 2) if recover == recover else np.nan,
        "cte_hash": round(float(np.abs(cte).sum()), 6),
    })

    key = (method, round(offset, 1))
    if key not in traces:
        traces[key] = (t, cte * 1000, err, corr)

if not rows:
    raise SystemExit(f"No usable baseline CSVs in {RES}")

df = pd.DataFrame(rows).sort_values(["Offset_mm", "Method", "Run"])
df.to_csv(OUT / "diagnostics.csv", index=False)

print("=" * 78)
print(" PER-RUN DIAGNOSTICS")
print("=" * 78)
print(df.drop(columns=["cte_hash"]).to_string(index=False))

# ------------------------------------------------------------------ verdicts
print()
print("=" * 78)
print(" VERDICTS")
print("=" * 78)

issues = 0

# 1. saturation
worst = df.loc[df["Saturated_%"].idxmax()]
mean_sat = df["Saturated_%"].mean()
print(f"\n1. SATURATION   mean {mean_sat:.1f} % of frames, "
      f"worst {worst['Saturated_%']:.1f} % "
      f"({worst['Method']} @ {worst['Offset_mm']:+.0f} mm)")
if mean_sat > 25:
    issues += 1
    print("   PROBLEM. The error signal spends much of the run pinned at")
    print("   +/-1.0. In that regime P, PD and the CNN all emit the same")
    print("   command, which is exactly why your three methods agree to")
    print("   within 0.2 mm. Reduce the start offset until the line stays")
    print("   inside the camera's field of view for the whole run.")
elif mean_sat > 8:
    print("   BORDERLINE. Some saturation, mostly in the first seconds.")
    print("   Check the transient plot before drawing conclusions about")
    print("   overshoot or settling time.")
else:
    print("   OK. The controller is operating in its proportional region.")

# 2. clamping
mean_clamp = df["Clamped_%"].mean()
print(f"\n2. CLAMPING     mean {mean_clamp:.1f} % of frames at "
      f"+/-{MAX_CORRECTION}")
if mean_clamp > 15:
    issues += 1
    print("   PROBLEM. The correction is hitting MAX_CORRECTION often, so")
    print("   the gains you report are not the gains the robot experienced.")
else:
    print("   OK.")

# 3. dead signal
low = df[df["err_std"] < 0.02]
print(f"\n3. ERROR SIGNAL mean std {df['err_std'].mean():.4f}, "
      f"range {df['err_min'].min():+.2f} to {df['err_max'].max():+.2f}")
if len(low):
    issues += 1
    print("   PROBLEM. perceived_error barely varies in "
          f"{len(low)} run(s). Perception is not resolving lateral")
    print("   position - check DARK_THRESHOLD and the ROI ratio.")
else:
    print("   OK. Perception responds to the robot's position.")

# 4. determinism
dupes = (df.groupby(["Method", "Offset_mm"])["cte_hash"]
         .nunique().rename("distinct_trajectories"))
reps = df.groupby(["Method", "Offset_mm"]).size().rename("runs")
det = pd.concat([reps, dupes], axis=1)
det["identical"] = det["runs"] > det["distinct_trajectories"]
print("\n4. REPEATABILITY")
print(det.to_string())
if det["identical"].any():
    print("   The simulator is deterministic: repeat runs of the same")
    print("   condition reproduce the same trajectory exactly, which is why")
    print("   your standard deviations are 0.000. Do NOT present that as")
    print("   'mean +/- SD over three trials' - an examiner will read it as")
    print("   three independent trials when it is one trial run three")
    print("   times. Report a single value per condition and get your")
    print("   spread from varying the CONDITION instead: sweep the start")
    print("   offset (-30, -20, -10, 0, +10, +20, +30 mm) and report the")
    print("   mean and spread across offsets.")

# ------------------------------------------------------------------ plots
keys = [k for k in sorted(traces, key=lambda k: (k[1], ORDER.index(k[0])
                                                 if k[0] in ORDER else 9))]
offsets = sorted({k[1] for k in keys})

fig, axes = plt.subplots(2, len(offsets), figsize=(6 * len(offsets), 8),
                         squeeze=False)
for col, off in enumerate(offsets):
    for method in ORDER:
        if (method, off) not in traces:
            continue
        t, cte_mm, err, corr = traces[(method, off)]
        axes[0][col].plot(t, cte_mm, label=method)
        axes[1][col].plot(t, err, label=method)
    axes[0][col].axhline(0, c="k", lw=0.8)
    axes[0][col].set_title(f"Cross-track error, start {off:+.0f} mm")
    axes[0][col].set_ylabel("CTE (mm)")
    axes[1][col].axhline(1.0, ls=":", c="r", lw=1)
    axes[1][col].axhline(-1.0, ls=":", c="r", lw=1)
    axes[1][col].set_title("Perceived error (red = saturation)")
    axes[1][col].set_ylabel("perceived_error")
    axes[1][col].set_ylim(-1.2, 1.2)
    for a in (axes[0][col], axes[1][col]):
        a.set_xlabel("Time (s)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "diagnostics.png", dpi=150)
plt.close()

print(f"\nSaved -> {OUT / 'diagnostics.csv'}")
print(f"Saved -> {OUT / 'diagnostics.png'}")
if issues:
    print(f"\n{issues} blocking issue(s) found - fix these before writing "
          f"the comparison chapter.")
