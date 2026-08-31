"""
cnn_line_follower_v2.py
=======================
V2 Webots CNN controller.

Changes:
- lower-road ROI only
- predicts actual PD correction
- faster steering response
- no classical image processing or PD calculation during inference
"""

from controller import Robot
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28
BASE_SPEED = 1.20

MODEL_W = 80
MODEL_H = 45

# Smaller value = faster reaction.
SMOOTHING = 0.25

# Safety limit for predicted correction.
MAX_CORRECTION = 2.0

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

CAM_W = camera.getWidth()
CAM_H = camera.getHeight()

HERE = Path(__file__).resolve().parent
CONTROLLERS = HERE.parent

candidates = [
    CONTROLLERS / "pd_line_follower" / "cnn_output_v2" / "best_model_v2.keras",
    HERE / "best_model_v2.keras",
]

MODEL_PATH = next((p for p in candidates if p.exists()), None)

if MODEL_PATH is None:
    print("ERROR: best_model_v2.keras not found.")
    for p in candidates:
        print(" -", p)
    sys.exit(1)

print("\n============================================")
print(" CNN LINE FOLLOWER V2")
print("============================================")
print("Model:", MODEL_PATH)
print("ROI  : lower 45% of camera")
print("============================================\n")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("[CNN V2] Model loaded successfully.")
print("[CNN V2] Starting autonomous CNN control...")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def frame_for_model():
    raw = camera.getImage()
    if raw is None:
        return None

    arr = np.frombuffer(raw, dtype=np.uint8)
    expected = CAM_W * CAM_H * 4
    if arr.size != expected:
        return None

    bgra = arr.reshape((CAM_H, CAM_W, 4))
    rgb = bgra[:, :, [2, 1, 0]]

    # Lower 45% ROI, exactly matching V2 training.
    y0 = int(CAM_H * 0.55)
    roi = rgb[y0:CAM_H, :, :]

    img = tf.convert_to_tensor(roi, dtype=tf.float32)
    img = tf.image.resize(img, [MODEL_H, MODEL_W], method="bilinear")
    img = img / 255.0
    img = tf.expand_dims(img, 0)

    return img

previous_correction = 0.0
step_count = 0

while robot.step(TIME_STEP) != -1:
    step_count += 1

    image = frame_for_model()

    if image is None:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        continue

    pred = model(image, training=False)
    correction = float(pred.numpy().reshape(-1)[0])

    correction = clamp(correction, -MAX_CORRECTION, MAX_CORRECTION)

    correction = (
        SMOOTHING * previous_correction
        + (1.0 - SMOOTHING) * correction
    )
    previous_correction = correction

    # Same wheel mapping as the teacher PD controller.
    left_speed = BASE_SPEED + correction
    right_speed = BASE_SPEED - correction

    left_speed = clamp(left_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    if step_count % 20 == 0:
        print(
            f"[CNN V2] correction={correction:+.4f} "
            f"left={left_speed:+.3f} right={right_speed:+.3f}"
        )
