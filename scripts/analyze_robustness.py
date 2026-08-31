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
DARK_THRESHOLD = 90.0          # must match common/line_vision.py


def contrast_of(world):
    if world.startswith("robust_c"):
        digits = world[len("robust_c"):]
        try:
            return int(digits) / 1000.0
        except ValueError:
            return float("nan")
    return float("nan")


rows = []
skipped = []

for f in sorted(RES.glob("*__*_run_*.csv")):
    head = f.stem.rsplit("_run_", 1)[0]
    if "__" not in head:
        continue
    method_tag, world = head.split("__", 1)

    if not world.startswith("robust_c"):
        continue                       # baseline sweep -> analyze_results.py

    d = pd.read_csv(f)
    if d.empty:
        skipped.append((f.name, "empty"))
        continue

    need = ["cte_m", "progress", "line_visible"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        skipped.append((f.name, f"missing {missing}"))
        continue

    e = pd.to_numeric(d["cte_m"], errors="coerce").dropna().to_numpy(float)
    if e.size == 0:
        skipped.append((f.name, "no numeric cte_m"))
        continue

    if "line_dark_p2" in d.columns:
        grey = float(pd.to_numeric(d["line_dark_p2"], errors="coerce").mean())
    else:
        grey = float("nan")

    rows.append({
        "Method": LABEL.get(method_tag, method_tag),
        "World": world,
        "baseColor": contrast_of(world),
        "Line_grey": round(grey, 1),
        "Completion_%": round(float(pd.to_numeric(
            d["progress"], errors="coerce").max()) * 100, 2),
        "LineVisible_%": round(float(pd.to_numeric(
            d["line_visible"], errors="coerce").mean()) * 100, 2),
        "MAE_mm": round(np.abs(e).mean() * 1000, 3),
        "RMSE_mm": round(float(np.sqrt((e ** 2).mean())) * 1000, 3),
        "MaxAbs_mm": round(np.abs(e).max() * 1000, 3),
        "Run": f.stem,
    })

if skipped:
    print("Skipped files:")
    for name, why in skipped:
        print(f"  {name}: {why}")
    print()

if not rows:
    raise SystemExit(
        f"No robustness result CSVs in {RES}.\n"
        "Expected names like  cnnpd__robust_c0330_run_001.csv\n"
        "Run:  python scripts/make_experiment_worlds.py\n"
        "      python scripts/run_all_experiments.py --repeats 1 --only robust")

runs = pd.DataFrame(rows).sort_values(["baseColor", "Method"])
runs.to_csv(OUT / "robustness_runs.csv", index=False)

# ---- pivot on the controlled variable, not the measured one ----
pivot_done = runs.pivot_table(index="baseColor", columns="Method",
                              values="Completion_%", aggfunc="mean")
pivot_vis = runs.pivot_table(index="baseColor", columns="Method",
                             values="LineVisible_%", aggfunc="mean")

# one measured grey per condition, for the x-axis
grey_map = runs.groupby("baseColor")["Line_grey"].mean().round(1)

table = pivot_done.copy()
table.insert(0, "Line_grey", grey_map)
table.to_csv(OUT / "robustness_completion.csv")

vis_table = pivot_vis.copy()
vis_table.insert(0, "Line_grey", grey_map)
vis_table.to_csv(OUT / "robustness_visibility.csv")

print("--- per run ---")
print(runs.to_string(index=False))
print("\n--- track completion % vs contrast ---")
print(table.round(2).to_string())
print("\n--- classical detector line-visibility % vs contrast ---")
print(vis_table.round(2).to_string())

order = [m for m in ORDER if m in pivot_done.columns]
order += [m for m in pivot_done.columns if m not in order]

x = grey_map.reindex(pivot_done.index).to_numpy(float)
use_grey = np.isfinite(x).all() and len(set(x)) == len(x)
if not use_grey:
    x = pivot_done.index.to_numpy(float)
    xlabel = "Scene luminosity (1.00 = baseline)"
    
else:
    xlabel = "Scene luminosity (1.00 = baseline)"

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

for m in order:
    ax[0].plot(x, pivot_done[m].to_numpy(float), marker="o", label=m)
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel("Track completion (%)")
ax[0].set_title("Robustness to visual degradation")
ax[0].set_ylim(0, 105)

for m in order:
    ax[1].plot(x, pivot_vis[m].to_numpy(float), marker="s", label=m)
ax[1].axhline(50, ls="--", c="r", lw=1, label="50 % visibility")
ax[1].set_xlabel(xlabel)
ax[1].set_ylabel("Frames with line detected (%)")
ax[1].set_title("Fixed-threshold detector breakdown")
ax[1].set_ylim(0, 105)

if use_grey:
    for a in ax:
        a.axvline(DARK_THRESHOLD, ls=":", c="k", lw=1.2,
                  label=f"DARK_THRESHOLD = {DARK_THRESHOLD:.0f}")

for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "robustness_sweep.png", dpi=160)
plt.close()

print(f"\nSaved -> {OUT}")

# ---- where does the classical detector break? ----
classical = [m for m in ("P", "PD") if m in pivot_vis.columns]
if classical:
    weak = pivot_vis[pivot_vis[classical].mean(axis=1) < 90.0]
    if len(weak):
        c = weak.index.min()
        print(f"\nClassical detector starts failing at baseColor >= {c:.3f} "
              f"(measured line grey {grey_map.get(c, float('nan')):.1f})")
    else:
        hi = pivot_vis.index.max()
        print(f"\nClassical detector never dropped below 90 % visibility.")
        print(f"Highest contrast tested: baseColor {hi:.3f}, measured line "
              f"grey {grey_map.get(hi, float('nan')):.1f}, versus a threshold "
              f"of {DARK_THRESHOLD:.0f}.")
        if np.isfinite(grey_map.get(hi, float("nan"))) \
                and grey_map[hi] < DARK_THRESHOLD:
            print("The line is still darker than the threshold, so the "
                  "detector was never stressed - raise CONTRASTS in "
                  "scripts/make_experiment_worlds.py.")
        else:
            print("The line is already lighter than the threshold, so if "
                  "visibility stayed high the detector is reading something "
                  "other than the painted line - inspect a camera frame.")