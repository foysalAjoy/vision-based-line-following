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
SETTLE_BAND_MM = 3.0        # residual band counted as settled


def load_all():
    out = []
    for f in sorted(RES.glob("*__*_run_*.csv")):
        head = f.stem.rsplit("_run_", 1)[0]
        method_tag, world = head.split("__", 1)
        d = pd.read_csv(f)
        if d.empty:
            continue
        out.append({
            "method": LABEL.get(method_tag, method_tag),
            "world": world,
            "start_cte": float(d["cte_m"].iloc[0]),
            "t": d["time_s"].to_numpy(dtype=float),
            "cte": d["cte_m"].to_numpy(dtype=float),
            "file": f.stem,
        })
    return out


runs = load_all()
if not runs:
    raise SystemExit(f"No tagged CSVs in {RES}")

centred = {(r["method"], r["world"]): r
           for r in runs if abs(r["start_cte"]) < 0.005}
offsets = [r for r in runs if abs(r["start_cte"]) >= 0.005]

if not centred:
    raise SystemExit("No centred runs found. Run each controller once with "
                     "start_offset_y=0.0 before running this script.")
if not offsets:
    raise SystemExit("No offset runs found. Run each controller with "
                     "start_offset_y=0.030.")

rows = []
curves = []
unpaired = []

for r in offsets:
    ref = centred.get((r["method"], r["world"]))
    if ref is None:
        unpaired.append(f"{r['method']}/{r['world']}")
        continue

    # resample the centred run onto the offset run's time base
    ref_cte = np.interp(r["t"], ref["t"], ref["cte"])
    resid = r["cte"] - ref_cte

    sign0 = 1.0 if resid[0] >= 0 else -1.0
    norm = resid * sign0                    # start is now positive

    peak_opposite = float(max(0.0, (-norm).max()))
    overshoot_pct = 100.0 * peak_opposite / abs(resid[0]) if resid[0] else 0.0

    band = SETTLE_BAND_MM / 1000.0
    settle_s = float("nan")
    for i in range(resid.size):
        if np.all(np.abs(resid[i:]) <= band):
            settle_s = float(r["t"][i])
            break

    # time to first reach 10 % of the initial offset
    rise_s = float("nan")
    thresh = 0.10 * abs(resid[0])
    hit = np.nonzero(np.abs(resid) <= thresh)[0]
    if hit.size:
        rise_s = float(r["t"][hit[0]])

    rows.append({
        "Method": r["method"],
        "World": r["world"],
        "InitialOffset_mm": round(abs(resid[0]) * 1000, 2),
        "Overshoot_mm": round(peak_opposite * 1000, 3),
        "Overshoot_%": round(overshoot_pct, 2),
        "RiseTime_s": round(rise_s, 3),
        f"Settle{int(SETTLE_BAND_MM)}mm_s": round(settle_s, 3),
        "ResidualRMS_mm": round(np.sqrt((resid ** 2).mean()) * 1000, 3),
        "Run": r["file"],
    })
    curves.append((r["method"], r["t"], resid * 1000))

if unpaired:
    uniq = sorted(set(unpaired))
    print(f"  {len(unpaired)} offset run(s) had no centred reference "
          f"- skipped.")
    print(f"  ({', '.join(uniq)})")
    print("  This is expected for the robustness sweep, which is only "
          "run at one offset.")

if not rows:
    raise SystemExit("No offset run could be paired with a centred run.")

table = pd.DataFrame(rows).sort_values(["World", "Method"])
table.to_csv(OUT / "transient_response.csv", index=False)

print("\n--- offset recovery (curve error removed) ---")
print(table.to_string(index=False))

plt.figure(figsize=(10, 5))
for method, t, resid_mm in curves:
    plt.plot(t, resid_mm, label=method)
plt.axhline(0, c="k", lw=0.8)
plt.axhline(SETTLE_BAND_MM, ls=":", c="grey", lw=0.8)
plt.axhline(-SETTLE_BAND_MM, ls=":", c="grey", lw=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Transient cross-track error (mm)")
plt.title("Offset recovery with curve-following error removed")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "transient_response.png", dpi=160)
plt.close()

print(f"\nSaved -> {OUT}")