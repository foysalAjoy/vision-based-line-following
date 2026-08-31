import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "cnn_dataset" / "prepared_v1"
CSV = PREP / "manifest_line_error.csv"
IMAGES = PREP / "images"
MODEL = ROOT / "models_trained" / "line_error_cnn.keras"
OUT = ROOT / "evaluation_summary_final"
OUT.mkdir(parents=True, exist_ok=True)

# must match train_line_error_cnn.py exactly
SEED = 42
IMG_W, IMG_H = 160, 96
VAL_FRAC = 0.20

for p in (CSV, MODEL):
    if not p.exists():
        sys.exit(f"ERROR: missing {p}")

import tensorflow as tf

df = pd.read_csv(CSV)

groups = df["group_id"].unique()
rng = np.random.default_rng(SEED)
rng.shuffle(groups)
n_val = max(1, int(len(groups) * VAL_FRAC))
val_groups = set(groups[:n_val])

train_df = df[~df["group_id"].isin(val_groups)]
val_df = df[df["group_id"].isin(val_groups)].reset_index(drop=True)
assert not (set(train_df["group_id"]) & set(val_df["group_id"])), "leak!"


def load_gray(name):
    a = Image.open(IMAGES / str(name)).convert("L")
    if a.size != (IMG_W, IMG_H):
        a = a.resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(a, dtype=np.uint8)


x_val = np.stack([load_gray(n) for n in val_df["image_filename"]])[..., None]
y_val = val_df["line_error"].to_numpy(dtype=np.float32)

model = tf.keras.models.load_model(str(MODEL), compile=False)
pred = model.predict(x_val.astype(np.float32) / 255.0,
                     verbose=0).reshape(-1)

resid = pred - y_val
mae = float(np.abs(resid).mean())
rmse = float(np.sqrt((resid ** 2).mean()))
ss_res = float((resid ** 2).sum())
ss_tot = float(((y_val - y_val.mean()) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
baseline = float(np.abs(y_val - float(train_df["line_error"].mean())).mean())

off = np.abs(y_val) > 0.10
sign_acc = float(((pred > 0) == (y_val > 0))[off].mean() * 100) if off.any() \
    else float("nan")

stats = pd.DataFrame([{
    "Validation frames": len(y_val),
    "Validation groups": val_df["group_id"].nunique(),
    "MAE": round(mae, 5),
    "RMSE": round(rmse, 5),
    "R2": round(r2, 4),
    "Sign accuracy (%)": round(sign_acc, 2),
    "Constant-predictor MAE": round(baseline, 5),
    "Improvement over baseline (%)": round((1 - mae / baseline) * 100, 1),
    "Prediction std": round(float(pred.std()), 4),
    "Target std": round(float(y_val.std()), 4),
}])
stats.to_csv(OUT / "cnn_validation.csv", index=False)

print("=" * 60)
print(" CNN VALIDATION")
print("=" * 60)
for c in stats.columns:
    print(f"  {c:<32} {stats.iloc[0][c]}")
print("=" * 60)
if pred.std() < 0.5 * y_val.std():
    print(" WARNING: predictions are far less spread than the targets -")
    print(" the network is regressing towards the mean.")

# ---------------- scatter ----------------
plt.figure(figsize=(6.4, 6.0))
plt.scatter(y_val, pred, s=14, alpha=0.55, edgecolors="none", c="#3377bb")
lim = 1.05
plt.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal (y = x)")
plt.axhline(0, c="grey", lw=0.6)
plt.axvline(0, c="grey", lw=0.6)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.xlabel("True normalised line error")
plt.ylabel("Predicted normalised line error")
plt.title(f"CNN validation: predicted vs true\n"
          f"MAE {mae:.4f}   $R^2$ {r2:.3f}   sign accuracy {sign_acc:.1f} %")
plt.legend()
plt.grid(alpha=0.3)
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT / "cnn_pred_vs_true.png", dpi=160)
plt.close()

# ---------------- residual histogram ----------------
plt.figure(figsize=(7.2, 4.2))
plt.hist(resid, bins=40, color="#33aa77", edgecolor="black")
plt.axvline(0, c="k", lw=1)
plt.xlabel("Prediction error (predicted - true)")
plt.ylabel("Validation frames")
plt.title("CNN prediction error distribution")
plt.grid(axis="y", alpha=0.3)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / "cnn_error_hist.png", dpi=160)
plt.close()

print(f"\nSaved -> {OUT / 'cnn_pred_vs_true.png'}")
print(f"Saved -> {OUT / 'cnn_error_hist.png'}")
print(f"Saved -> {OUT / 'cnn_validation.csv'}")
