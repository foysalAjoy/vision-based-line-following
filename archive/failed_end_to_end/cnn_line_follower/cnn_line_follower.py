"""
cnn_line_follower.py
====================

Webots CNN-based line-following controller.

This controller:
- loads the trained best_model.keras
- reads the e-puck camera
- preprocesses frames exactly like training
- predicts one continuous steering value
- converts steering into left/right wheel speeds
- drives autonomously WITHOUT the classical PD controller

Expected project layout:

vision_line_following/
    controllers/
        pd_line_follower/
            cnn_output/
                best_model.keras

        cnn_line_follower/
            cnn_line_follower.py

The script automatically finds the trained model in the
pd_line_follower/cnn_output folder.
"""

from controller import Robot
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


# ============================================================
# WEBOTS SETUP
# ============================================================

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28
BASE_SPEED = 1.20

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# ============================================================
# CAMERA
# ============================================================

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

CAM_W = camera.getWidth()
CAM_H = camera.getHeight()

MODEL_W = 80
MODEL_H = 60


# ============================================================
# FIND + LOAD TRAINED MODEL
# ============================================================

HERE = Path(__file__).resolve().parent
CONTROLLERS_DIR = HERE.parent

candidate_models = [
    CONTROLLERS_DIR
    / "pd_line_follower"
    / "cnn_output"
    / "best_model.keras",

    HERE
    / "best_model.keras",

    HERE
    / "cnn_output"
    / "best_model.keras",
]

MODEL_PATH = None

for candidate in candidate_models:
    if candidate.exists():
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    print("")
    print("============================================")
    print(" ERROR: TRAINED CNN MODEL NOT FOUND")
    print("============================================")
    print("Expected one of:")
    for p in candidate_models:
        print(" -", p)
    print("============================================")
    sys.exit(1)

print("")
print("============================================")
print(" CNN LINE FOLLOWER")
print("============================================")
print(f"Camera        : {CAM_W} x {CAM_H}")
print(f"Model input   : {MODEL_W} x {MODEL_H}")
print(f"Model         : {MODEL_PATH}")
print(f"Base speed    : {BASE_SPEED}")
print("============================================")
print("")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("[CNN] Model loaded successfully.")


# ============================================================
# CONTROL SETTINGS
# ============================================================

# The training label was:
#
# steering = (right_speed - left_speed) / (2 * MAX_SPEED)
#
# Therefore reconstruction is:
#
# left  = base - steering * MAX_SPEED
# right = base + steering * MAX_SPEED
#
# Gain lets us slightly strengthen the learned steering if needed.
STEERING_GAIN = 1.0

# Smooth predictions to reduce jitter.
SMOOTHING = 0.65

previous_steering = 0.0

# Safety clamp.
MAX_STEERING = 0.45


def clamp(value, low, high):
    return max(low, min(high, value))


def get_model_input():
    """
    Read Webots BGRA camera image and convert to RGB float32 image
    with the exact 60x80 input size used during CNN training.
    """

    raw = camera.getImage()

    if raw is None:
        return None

    frame = np.frombuffer(raw, dtype=np.uint8)

    expected = CAM_W * CAM_H * 4

    if frame.size != expected:
        return None

    frame = frame.reshape((CAM_H, CAM_W, 4))

    # Webots gives BGRA.
    # Convert to RGB.
    rgb = frame[:, :, [2, 1, 0]]

    # TensorFlow resize is used because training used tf.image.resize.
    image = tf.convert_to_tensor(
        rgb,
        dtype=tf.float32
    )

    image = tf.image.resize(
        image,
        [MODEL_H, MODEL_W],
        method="bilinear"
    )

    image = image / 255.0

    # Add batch dimension:
    # (60, 80, 3) -> (1, 60, 80, 3)
    image = tf.expand_dims(image, axis=0)

    return image


print("[CNN] Starting autonomous CNN control...")
print("")


# ============================================================
# MAIN LOOP
# ============================================================

step_count = 0

while robot.step(TIME_STEP) != -1:

    step_count += 1

    image = get_model_input()

    if image is None:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        continue

    # CNN inference.
    prediction = model(
        image,
        training=False
    )

    steering = float(
        prediction.numpy().reshape(-1)[0]
    )

    # Strength / safety.
    steering *= STEERING_GAIN

    steering = clamp(
        steering,
        -MAX_STEERING,
        MAX_STEERING
    )

    # Smooth steering between frames.
    steering = (
        SMOOTHING * previous_steering
        + (1.0 - SMOOTHING) * steering
    )

    previous_steering = steering

    # Convert learned steering back into wheel speeds.
    left_speed = (
        BASE_SPEED
        - steering * MAX_SPEED
    )

    right_speed = (
        BASE_SPEED
        + steering * MAX_SPEED
    )

    left_speed = clamp(
        left_speed,
        -MAX_SPEED,
        MAX_SPEED
    )

    right_speed = clamp(
        right_speed,
        -MAX_SPEED,
        MAX_SPEED
    )

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    if step_count % 20 == 0:
        print(
            f"[CNN] steering={steering:+.4f} "
            f"left={left_speed:+.3f} "
            f"right={right_speed:+.3f}"
        )
