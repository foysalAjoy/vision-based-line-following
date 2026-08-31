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
BASELINE_WORLD = "baseline"

REQUIRED = ["time_s", "cte_m", "progress", "correction", "line_visible"]


def parse_name(stem):
    """cnnpd__baseline_run_001 -> ('cnnpd', 'baseline', 1)"""
    head, _, run = stem.rpartition("_run_")
    if not head:
        return None, None, None
    method_tag, _, world = head.partition("__")
    try:
        run_no = int(run)
    except ValueError:
        run_no = 0
    return method_tag, world, run_no


rows = []
skipped = []

for f in sorted(RES.glob("*_run_*.csv")):
    method_tag, world, run_no = parse_name(f.stem)
    if method_tag is None:
        continue
    if world != BASELINE_WORLD:
        continue                      # robustness sweep -> analyze_robustness

    d = pd.read_csv(f)
    missing = [c for c in REQUIRED if c not in d.columns]
    if d.empty or missing:
        skipped.append((f.name, "empty" if d.empty else f"missing {missing}"))
        continue

    e = pd.to_numeric(d["cte_m"], errors="coerce").dropna().to_numpy(float)
    if e.size == 0:
        skipped.append((f.name, "no numeric cte_m - re-run the controller"))
        continue

    corr = pd.to_numeric(d["correction"], errors="coerce").dropna().to_numpy(float)
    offset = 0.0
    if "start_offset_mm" in d.columns:
        offset = float(pd.to_numeric(d["start_offset_mm"],
                                     errors="coerce").iloc[0])

    rows.append({
        "Method": LABEL.get(method_tag, method_tag),
        "Run": f.stem,
        "RunNo": run_no,
        "StartOffset_mm": round(offset, 1),
        "Samples": len(e),
        "Duration_s": round(float(pd.to_numeric(d["time_s"],
                                                errors="coerce").max()), 3),
        "Completion_%": round(float(pd.to_numeric(d["progress"],
                                                  errors="coerce").max()) * 100, 2),
        "MAE_mm": round(np.abs(e).mean() * 1000, 3),
        "RMSE_mm": round(float(np.sqrt((e ** 2).mean())) * 1000, 3),
        "MaxAbs_mm": round(np.abs(e).max() * 1000, 3),
        "LineVisible_%": round(float(pd.to_numeric(d["line_visible"],
                                                   errors="coerce").mean()) * 100, 2),
        "Smoothness": round(float(np.abs(np.diff(corr)).mean()), 5)
        if corr.size > 1 else np.nan,
    })

if skipped:
    print("Skipped files:")
    for name, why in skipped:
        print(f"  {name}: {why}")
    print()

if not rows:
    raise SystemExit(
        f"No usable baseline result CSVs in {RES}.\n"
        "Run the three controllers on worlds/line_following_curve_test.wbt "
        "first.")

runs = pd.DataFrame(rows)
runs = runs.sort_values(["Method", "StartOffset_mm", "RunNo"])
runs.to_csv(OUT / "all_runs.csv", index=False)

# ---------------------------------------------------------------- headline
# Webots is deterministic, so repeating one world gives SD = 0.000 and is
# not replication. When the run set contains a SWEEP of start offsets, the
# headline table aggregates across those offsets instead: the spread then
# reflects how sensitive each method is to its initial condition, which is
# a real and defensible source of variation.
nonzero = sorted({o for o in runs["StartOffset_mm"].unique() if abs(o) > 1e-6})

if len(nonzero) >= 3:
    main = runs[runs["StartOffset_mm"].abs() > 1e-6]
    spread_source = f"across {len(nonzero)} start offsets " \
                    f"({min(nonzero):+.0f} to {max(nonzero):+.0f} mm)"
elif nonzero:
    main_offset = max(nonzero, key=abs)
    main = runs[runs["StartOffset_mm"] == main_offset]
    spread_source = f"at a single start offset of {main_offset:+.1f} mm"
else:
    main = runs
    spread_source = "at zero start offset"

# Deterministic duplicates inside one condition add no information, so
# collapse them before averaging or the SD is understated.
before = len(main)
main = main.drop_duplicates(subset=["Method", "StartOffset_mm", "MAE_mm",
                                    "RMSE_mm", "Smoothness"])
collapsed = before - len(main)

print(f"Headline table: spread measured {spread_source}; "
      f"{len(main)} distinct runs"
      + (f" ({collapsed} identical repeat(s) collapsed)" if collapsed else ""))
if len(nonzero) < 3:
    print("  NOTE: repeats of one identical world are deterministic and will")
    print("        report SD = 0.000. Generate the offset sweep with")
    print("        scripts/make_experiment_worlds.py for meaningful spread.")

metrics = ["MAE_mm", "RMSE_mm", "MaxAbs_mm", "Completion_%",
           "LineVisible_%", "Smoothness"]

mean = main.groupby("Method")[metrics].mean().round(3)
sd = main.groupby("Method")[metrics].std(ddof=1).round(3).fillna(0.0)
count = main.groupby("Method").size().rename("N")

agg = mean.copy()
for m in metrics:
    agg[m] = [f"{mu:.3f} ± {s:.3f}" for mu, s in zip(mean[m], sd[m])]
agg = agg.join(count)

present = [m for m in ORDER if m in agg.index]
present += [m for m in agg.index if m not in present]
agg = agg.loc[present].reset_index()
agg.to_csv(OUT / "method_comparison.csv", index=False)

mean_plot = mean.loc[present]

print()
print(runs.to_string(index=False))
print()
print(agg.to_string(index=False))

# ---------------- figure 1: accuracy + smoothness ----------------
colors = ["#8a8a8a", "#33aa77", "#3377bb"][:len(present)]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].bar(present, mean_plot["MAE_mm"],
          yerr=sd.loc[present, "MAE_mm"], capsize=4, color=colors)
ax[0].set_ylabel("Mean cross-track error (mm)")
ax[0].set_title("Tracking accuracy (lower = better)")
ax[1].bar(present, mean_plot["Smoothness"],
          yerr=sd.loc[present, "Smoothness"], capsize=4, color=colors)
ax[1].set_ylabel("Mean |Δcorrection| per step")
ax[1].set_title("Control smoothness (lower = better)")
for a in ax:
    a.grid(axis="y", alpha=0.3)
    a.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / "method_comparison.png", dpi=160)
plt.close()

# ---------------- figure 2: error over time ----------------
# Trace one representative run per method: the largest positive offset
# present, so all three curves start from the same initial condition.
pos = [o for o in main["StartOffset_mm"].unique() if o > 1e-6]
trace_offset = max(pos) if pos else float(main["StartOffset_mm"].iloc[0])

plt.figure(figsize=(10, 5))
plotted = 0
for method in present:
    sub = main[(main["Method"] == method)
               & (main["StartOffset_mm"] == trace_offset)].sort_values("RunNo")
    if sub.empty:
        sub = main[main["Method"] == method].sort_values("RunNo")
    if sub.empty:
        continue
    f = RES / f"{sub.iloc[0]['Run']}.csv"
    d = pd.read_csv(f)
    plt.plot(pd.to_numeric(d["time_s"], errors="coerce"),
             pd.to_numeric(d["cte_m"], errors="coerce") * 1000, label=method)
    plotted += 1

if plotted:
    plt.axhline(0, c="k", lw=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Cross-track error (mm)")
    plt.title(f"Ground-truth tracking error ({trace_offset:+.0f} mm start)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "error_over_time.png", dpi=160)
plt.close()

print(f"\nSaved -> {OUT}")
