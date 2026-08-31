import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "cnn_dataset" / "prepared_v1"
CSV = PREP / "manifest_line_error.csv"
IMAGES = PREP / "images"

OUT = ROOT / "models_trained"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = OUT / "line_error_cnn.keras"

IMG_W, IMG_H = 160, 96
BATCH = 32
EPOCHS = 60
VAL_FRAC = 0.20
SHIFTS = [-24, -16, -8, 0, 8, 16, 24]     # pixels, in the 160-wide image

if not CSV.exists():
    sys.exit(f"ERROR: {CSV} missing.\nRun: python scripts/relabel_prepared_dataset.py")

df = pd.read_csv(CSV)

# ---------- group-aware split (no original/mirror leakage) ----------
groups = df["group_id"].unique()
rng = np.random.default_rng(SEED)
rng.shuffle(groups)
n_val = max(1, int(len(groups) * VAL_FRAC))
val_groups = set(groups[:n_val])

train_df = df[~df["group_id"].isin(val_groups)].reset_index(drop=True)
val_df = df[df["group_id"].isin(val_groups)].reset_index(drop=True)

assert not (set(train_df["group_id"]) & set(val_df["group_id"])), "leak!"


def load_gray(name):
    a = Image.open(IMAGES / str(name)).convert("L")
    if a.size != (IMG_W, IMG_H):
        a = a.resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(a, dtype=np.uint8)


def shift_image(img, s):
    """Shift contents horizontally, fill the exposed side with floor white."""
    out = np.full_like(img, 234)
    if s == 0:
        return img.copy()
    if s > 0:
        out[:, s:] = img[:, :-s]
    else:
        out[:, :s] = img[:, -s:]
    return out


IC = (IMG_W - 1) / 2.0

# ---------- training set: real frames + synthetic lateral shifts ----------
xs, ys = [], []
for _, r in train_df.iterrows():
    base = load_gray(r["image_filename"])
    for s in SHIFTS:
        target = float(np.clip(r["line_error"] + s / IC, -1.0, 1.0))
        xs.append(shift_image(base, s))
        ys.append(target)

x_train = np.stack(xs)[..., None]
y_train = np.asarray(ys, dtype=np.float32)

# ---------- validation set: real frames only, no augmentation ----------
x_val = np.stack([load_gray(n) for n in val_df["image_filename"]])[..., None]
y_val = val_df["line_error"].to_numpy(dtype=np.float32)

baseline_mae = float(np.mean(np.abs(y_val - y_train.mean())))

print("")
print("=" * 52)
print(" LINE-ERROR CNN DATA")
print("=" * 52)
print(f" train groups / images : {train_df['group_id'].nunique()} / {len(train_df)}")
print(f" after shift augment   : {len(x_train)}")
print(f" val   groups / images : {val_df['group_id'].nunique()} / {len(val_df)}")
print(f" train target std      : {y_train.std():.4f}")
print(f" val   target std      : {y_val.std():.4f}")
print(f" constant-predictor MAE: {baseline_mae:.4f}   <-- CNN must beat this")
print("=" * 52)
print("")


def to_ds(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b),
                num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        def photometric(a, b):
            # Simulate the contrast sweep: lift the dark line towards the
            # floor and compress the dynamic range. Without this the CNN
            # only ever sees baseColor 0.01 and inherits its illumination.
            a = tf.image.random_brightness(a, max_delta=0.35)
            a = tf.image.random_contrast(a, lower=0.35, upper=1.30)
            gamma = tf.random.uniform([], 0.55, 1.60)
            a = tf.pow(tf.clip_by_value(a, 1e-4, 1.0), gamma)
            a = a + tf.random.normal(tf.shape(a), stddev=0.02)
            return tf.clip_by_value(a, 0.0, 1.0), b

        ds = ds.map(photometric, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# 96x160 -> 48x80 -> 24x40 -> 12x20 -> 6x10  (4 x stride-2 convs)
# Then pool over HEIGHT ONLY (6 -> 1) so the horizontal axis survives.
# GlobalAveragePooling2D would collapse the width too, which makes the
# network translation-invariant and therefore unable to locate the line.
model = tf.keras.Sequential([
    tf.keras.layers.Input((IMG_H, IMG_W, 1)),
    tf.keras.layers.Conv2D(16, 5, strides=2, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(48, 3, strides=2, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),

    tf.keras.layers.AveragePooling2D(pool_size=(6, 1)),   # (6,10,64)->(1,10,64)
    tf.keras.layers.Flatten(),                            # -> 640 features
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="tanh", name="line_error"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(8e-4),
    loss=tf.keras.losses.Huber(delta=0.1),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
)
model.summary()

history = model.fit(
    to_ds(x_train, y_train, True),
    validation_data=to_ds(x_val, y_val, False),
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(MODEL), monitor="val_mae",
                                           mode="min", save_best_only=True,
                                           verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min",
                                         patience=12,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min",
                                             factor=0.5, patience=5,
                                             min_lr=1e-6, verbose=1),
    ],
    verbose=1,
)

hist = pd.DataFrame(history.history)
hist.to_csv(OUT / "training_history.csv", index=False)

for key, title, fname in (("loss", "Huber loss", "training_loss.png"),
                          ("mae", "MAE", "training_mae.png")):
    plt.figure(figsize=(8, 5))
    plt.plot(hist[key], label="train")
    plt.plot(hist["val_" + key], label="validation")
    if key == "mae":
        plt.axhline(baseline_mae, ls="--", c="r", label="constant baseline")
    plt.xlabel("Epoch"); plt.ylabel(title); plt.title(f"Line-error CNN {title}")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / fname, dpi=160); plt.close()

pred = model.predict(x_val.astype(np.float32) / 255.0, verbose=0).reshape(-1)
val_mae = float(np.mean(np.abs(pred - y_val)))

print("")
print("=" * 52)
print(" TRAINING COMPLETE")
print("=" * 52)
print(f" validation MAE      : {val_mae:.5f}")
print(f" constant baseline   : {baseline_mae:.5f}")
print(f" improvement         : {(1 - val_mae / baseline_mae) * 100:.1f} %")
print(f" prediction std      : {pred.std():.4f}  (target std {y_val.std():.4f})")
print(f" model -> {MODEL}")
print("=" * 52)
if val_mae > baseline_mae * 0.7:
    print(" WARNING: the CNN is barely beating a constant predictor.")
    print(" Collect more non-centre frames before trusting the hybrid.")

    # ---- sign test: can the model tell left from right at all? ----
left_mask = y_val < -0.10
right_mask = y_val > 0.10
if left_mask.any() and right_mask.any():
    print("")
    print(" SIGN TEST")
    print(f"  true LEFT  (n={left_mask.sum():3d})  mean prediction "
          f"{pred[left_mask].mean():+.4f}   (should be clearly negative)")
    print(f"  true RIGHT (n={right_mask.sum():3d})  mean prediction "
          f"{pred[right_mask].mean():+.4f}   (should be clearly positive)")
    correct = ((pred > 0) == (y_val > 0))[np.abs(y_val) > 0.10]
    print(f"  sign accuracy on off-centre frames: {correct.mean() * 100:.1f} %")