import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "cnn_dataset" / "prepared_v1"
IMAGES = PREP / "images"
IN_CSV = PREP / "manifest.csv"
OUT_CSV = PREP / "manifest_line_error.csv"

DARK_THRESHOLD = 90.0
MIN_DARK_FRAC = 0.005
MAX_DARK_FRAC = 0.55

if not IN_CSV.exists():
    sys.exit(f"ERROR: {IN_CSV} not found")

df = pd.read_csv(IN_CSV)
rows, dropped = [], 0

for _, r in df.iterrows():
    path = IMAGES / str(r["image_filename"])
    if not path.exists():
        dropped += 1
        continue

    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    mask = a < DARK_THRESHOLD
    frac = float(mask.mean())

    if not (MIN_DARK_FRAC <= frac <= MAX_DARK_FRAC):
        dropped += 1
        continue

    cols = mask.sum(axis=0).astype(np.float32)
    total = float(cols.sum())
    x = np.arange(a.shape[1], dtype=np.float32)
    cx = float((x * cols).sum() / total)
    ic = (a.shape[1] - 1) / 2.0

    rows.append({
        "image_filename": r["image_filename"],
        "group_id": r["group_id"],
        "category": r["category"],
        "line_error": float(np.clip((cx - ic) / ic, -1.0, 1.0)),
        "dark_fraction": frac,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

print("=" * 52)
print(" RELABEL COMPLETE")
print("=" * 52)
print(f" input rows   : {len(df)}")
print(f" kept         : {len(out)}")
print(f" dropped      : {dropped}")
print(f" unique groups: {out['group_id'].nunique()}")
print(f" error  mean  : {out['line_error'].mean():+.4f}")
print(f" error  std   : {out['line_error'].std():.4f}")
print(f" error  range : {out['line_error'].min():+.4f} .. "
      f"{out['line_error'].max():+.4f}")
print(f" saved -> {OUT_CSV}")
print("=" * 52)