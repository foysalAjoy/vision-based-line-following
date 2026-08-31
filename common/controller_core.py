import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from controller import Supervisor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import line_vision as lv
import track_geometry as tg


# ---------------- shared tuning (identical for all methods) ----------------
BASE_SPEED = 2.60
MAX_SPEED = 6.28
MAX_CORRECTION = 2.20
DERIV_ALPHA = 0.30

START_GRACE_S = 1.0
LOST_FRAMES_TO_STOP = 15
SAFETY_TIMEOUT_S = 90.0

# ---------------- termination (perception-independent) --------------------
OFF_TRACK_M = 0.12
COMPLETION_TARGET = 0.995    
END_ZONE_PROGRESS = 0.90      
# ---------------- deterministic start pose (Supervisor) --------------------
START_TRANSLATION = [-0.62, 0.0, 0.0199]
START_ROTATION = [0.0, 0.0, 1.0, 0.0]
START_OFFSET_Y = 0.0
SETTLE_STEPS = 5

FLUSH_EVERY = 20      # rows

CSV_COLUMNS = [
    "time_s", "cte_m", "progress", "perceived_error",
    "p_term", "d_term", "correction", "left_speed", "right_speed",
    "line_visible", "dark_fraction", "roi_mean_gray", "line_dark_p2",
    "infer_ms", "pos_x", "pos_y", "start_offset_mm",
]


def _world_tag(robot):
    """
    Short identifier for the current world, used in the result filename.

        line_following_curve_test.wbt          -> baseline
        exp_baseline__pd_line_follower.wbt     -> baseline
        exp_robust_c0250__p_line_follower.wbt  -> robust_c0250
        robust_c0250.wbt                       -> robust_c0250

    The controller name is stripped so the same physical world always
    produces the same world tag no matter which method is driving.
    """
    try:
        stem = Path(robot.getWorldPath()).stem
    except Exception:
        return "world"
    if stem == "line_following_curve_test":
        return "baseline"
    if stem.startswith("exp_"):
        stem = stem[len("exp_"):]
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    return stem


def _cli_options(default_offset):
    """
    Overrides from Webots controllerArgs (sys.argv) or the environment.

        --offset=0.030     start_offset_y in metres
        --quit             quit Webots when the run finishes (batch mode)
        --tag=xyz          extra text appended to the result filename
    """
    offset = default_offset
    auto_quit = False
    tag = ""

    env_offset = os.environ.get("VLF_OFFSET")
    if env_offset:
        try:
            offset = float(env_offset)
        except ValueError:
            pass
    if os.environ.get("VLF_QUIT"):
        auto_quit = True

    for raw in sys.argv[1:]:
        arg = raw.strip()
        if not arg:
            continue
        if arg.startswith("--offset="):
            try:
                offset = float(arg.split("=", 1)[1])
            except ValueError:
                print(f"WARNING: could not parse {arg}, using {offset}")
        elif arg in ("--quit", "--batch"):
            auto_quit = True
        elif arg.startswith("--tag="):
            tag = arg.split("=", 1)[1].strip()

    return offset, auto_quit, tag


def run(method, kp, kd, model_path=None, start_offset_y=START_OFFSET_Y):
    start_offset_y, auto_quit, extra_tag = _cli_options(start_offset_y)

    robot = Supervisor()
    dt = int(robot.getBasicTimeStep())
    dt_s = dt / 1000.0

    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    for m in (left, right):
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    camera = robot.getDevice("downward camera")
    camera.enable(dt)
    cam_w, cam_h = camera.getWidth(), camera.getHeight()

    display = robot.getDevice("view")
    if display is None:
        print("WARNING: no Display device named 'view' in the .wbt - "
              "the camera overlay will not be shown")

    self_node = robot.getSelf()
    if self_node is None:
        print("ERROR: supervisor TRUE is not set on the E-puck in the .wbt")
        return

    # ---------------- force an identical start pose every run --------------
    trans_field = self_node.getField("translation")
    rot_field = self_node.getField("rotation")

    start_pos = list(START_TRANSLATION)
    start_pos[1] += start_offset_y

    trans_field.setSFVec3f(start_pos)
    rot_field.setSFRotation(list(START_ROTATION))
    self_node.resetPhysics()

    for _ in range(SETTLE_STEPS):
        if robot.step(dt) == -1:
            return
        left.setVelocity(0.0)
        right.setVelocity(0.0)

    # ---------------- optional CNN ----------------
    model = None
    if model_path is not None:
        import tensorflow as tf
        p = Path(model_path)
        if not p.exists():
            print(f"ERROR: CNN model not found -> {p}")
            print("Run: python scripts/train_line_error_cnn.py")
            return
        model = tf.keras.models.load_model(str(p), compile=False)
        model(np.zeros((1, lv.CNN_H, lv.CNN_W, 1), dtype=np.float32),
              training=False)
        print(f"[{method}] CNN loaded and warmed up: {p.name}")

    # ---------------- output ----------------
    root = Path(__file__).resolve().parents[1]
    world = _world_tag(robot)
    tag = method.lower().replace("+", "").replace(" ", "_")

    outdir = root / "evaluation_results_final"
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = f"{tag}__{world}"
    if extra_tag:
        prefix = f"{prefix}_{extra_tag}"
    n = len(list(outdir.glob(f"{prefix}_run_*.csv"))) + 1
    out_csv = outdir / f"{prefix}_run_{n:03d}.csv"

    fh = out_csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(CSV_COLUMNS)

    offset_mm = start_offset_y * 1000.0

    print("")
    print("=" * 52)
    print(f" {method}  |  Kp={kp}  Kd={kd}")
    print(f" world       : {world}")
    print(f" start offset: {offset_mm:+.1f} mm")
    print(f" perception  : {'CNN' if model is not None else 'classical'}")
    print(f" base speed  : {BASE_SPEED}   dt {dt} ms")
    print(f" log -> {out_csv.name}")
    print("=" * 52)

    prev_error = 0.0
    prev_deriv = 0.0
    last_good_error = 0.0
    lost = 0
    finished = False
    reason = ""
    rows_written = 0
    ctes, p_terms, d_terms = [], [], []
    latencies, grays, visibles, times = [], [], [], []
    max_progress = 0.0
    last_cte = 0.0
    t0 = robot.getTime()

    while robot.step(dt) != -1:
        t = robot.getTime() - t0

        if finished:
            left.setVelocity(0.0)
            right.setVelocity(0.0)
            continue

        bgr = lv.grab_bgr(camera, cam_w, cam_h)
        if bgr is None:
            continue

        gray = lv.roi_gray(bgr)
        roi_h, roi_w = gray.shape
        mean_gray = float(gray.mean())
        line_dark = float(np.percentile(gray, 2.0))   # ~darkest 2% = the line

        det_error, det_bbox, frac = lv.detect_line(gray)
        visible = det_error is not None

        # ---------------- perception source ----------------
        # The CNN path is NOT gated on the classical detector. If the
        # threshold detector fails, the CNN keeps producing an estimate and
        # the run is judged by where the robot actually ends up.
        cnn_box = None
        infer_ms = 0.0
        if model is None:
            error = det_error
        else:
            t_start = time.perf_counter()
            pred = model(lv.cnn_tensor(bgr), training=False)
            value = float(np.asarray(pred).reshape(-1)[0])
            infer_ms = (time.perf_counter() - t_start) * 1000.0
            latencies.append(infer_ms)

            error = lv.clamp(value, -1.0, 1.0)
            cnn_box = lv.error_to_box(error, roi_w, roi_h)

        # ---------------- termination ----------------
        if visible:
            lost = 0
        else:
            lost += 1

        if t > START_GRACE_S:   
            if max_progress >= COMPLETION_TARGET:
                finished, reason = True, "completed"
            elif abs(last_cte) > OFF_TRACK_M:
                finished, reason = True, "off_track"
            elif (model is None
                  and lost >= LOST_FRAMES_TO_STOP
                  and max_progress < END_ZONE_PROGRESS):   
                finished, reason = True, "line_lost"

        if not finished and t > SAFETY_TIMEOUT_S:
            finished, reason = True, "timeout"

        if finished:
            left.setVelocity(0.0)
            right.setVelocity(0.0)
            fh.close()
            _summary(method, world, start_offset_y, ctes, p_terms, d_terms,
                     latencies, grays, visibles, times, max_progress, t,
                     out_csv, root, reason)
            if auto_quit:
                robot.simulationQuit(0)
            continue

        # ---------------- PD control ----------------
        if error is None:
            error = last_good_error
        else:
            last_good_error = error

        raw_deriv = (error - prev_error) / dt_s
        filt_deriv = DERIV_ALPHA * raw_deriv + (1.0 - DERIV_ALPHA) * prev_deriv
        prev_deriv = filt_deriv
        prev_error = error

        p_term = kp * error
        d_term = kd * filt_deriv
        correction = lv.clamp(p_term + d_term,
                              -MAX_CORRECTION, MAX_CORRECTION)

        p_terms.append(p_term)
        d_terms.append(d_term)
        grays.append(mean_gray)
        visibles.append(1 if visible else 0)

        l_speed = lv.clamp(BASE_SPEED + correction, -MAX_SPEED, MAX_SPEED)
        r_speed = lv.clamp(BASE_SPEED - correction, -MAX_SPEED, MAX_SPEED)
        left.setVelocity(l_speed)
        right.setVelocity(r_speed)

        # ---------------- ground-truth metric ----------------
        px, py, _pz = self_node.getPosition()
        cte, progress = tg.cross_track_error(px, py)
        ctes.append(cte)
        times.append(t)
        last_cte = cte
        max_progress = max(max_progress, progress)

        writer.writerow([
            f"{t:.4f}",
            f"{cte:.6f}",
            f"{progress:.6f}",
            f"{error:.6f}",
            f"{p_term:.6f}",
            f"{d_term:.6f}",
            f"{correction:.6f}",
            f"{l_speed:.6f}",
            f"{r_speed:.6f}",
            1 if visible else 0,
            f"{frac:.6f}",
            f"{mean_gray:.4f}",
            f"{line_dark:.4f}",
            f"{infer_ms:.4f}",
            f"{px:.6f}",
            f"{py:.6f}",
            f"{offset_mm:.2f}",
        ])
        rows_written += 1
        if rows_written % FLUSH_EVERY == 0:
            fh.flush()

        # ---------------- display ----------------
        lv.render(
            display, bgr, roi_h,
            det_bbox=det_bbox,
            cnn_box=cnn_box,
            header=f"{method}  {world}  t={t:5.1f}s",
            lines=(
                f"err {error:+.3f}  P {p_term:+.3f}  D {d_term:+.3f}",
                f"CTE {cte * 1000:+6.1f} mm   prog {progress * 100:5.1f}%",
                f"dark {line_dark:.3f}  grey {mean_gray:5.1f}  vis {int(visible)}",
            ),
        )


def _summary(method, world, offset, ctes, p_terms, d_terms,
             latencies, grays, visibles, times, progress, elapsed,
             path, root, reason=""):
    a = np.abs(np.asarray(ctes, dtype=float))
    if a.size == 0:
        print(f"[{method}] no samples recorded")
        return

    p_abs = np.abs(np.asarray(p_terms, dtype=float))
    d_abs = np.abs(np.asarray(d_terms, dtype=float))
    p_mean = p_abs.mean() if p_abs.size else 0.0
    d_mean = d_abs.mean() if d_abs.size else 0.0
    ratio = (d_mean / p_mean * 100.0) if p_mean > 1e-12 else 0.0

    signed = np.asarray(ctes, dtype=float)
    t_arr = np.asarray(times, dtype=float)
    start_sign = 1.0 if signed[0] >= 0 else -1.0
    overshoot = float(max(0.0, (-start_sign * signed).max()))

    settle_band = 0.002
    settle_s = float("nan")
    for i in range(signed.size):
        if np.all(np.abs(signed[i:]) <= settle_band):
            settle_s = float(t_arr[i]) if i < t_arr.size else float("nan")
            break

    vis_pct = 100.0 * np.mean(visibles) if len(visibles) else 0.0
    grey_mean = float(np.mean(grays)) if len(grays) else float("nan")
    mae_mm = a.mean() * 1000.0
    rmse_mm = float(np.sqrt((a ** 2).mean())) * 1000.0
    max_mm = a.max() * 1000.0
    lat_mean = float(np.mean(latencies)) if latencies else float("nan")
    lat_p95 = float(np.percentile(latencies, 95)) if latencies else float("nan")

    print("")
    print("=" * 52)
    print(f" {method} - FINAL RESULT   [{world}]")
    print("=" * 52)
    print(f" Stopped because    : {reason or 'unknown'}")
    print(f" Samples            : {a.size}")
    print(f" Run time           : {elapsed:.2f} s")
    print(f" Start offset       : {offset * 1000:+.1f} mm")
    print(f" Track completion   : {progress * 100:.2f} %")
    print(f" Line visible       : {vis_pct:.2f} %")
    print(f" ROI mean grey      : {grey_mean:.1f}")
    print(f" MAE  cross-track   : {mae_mm:.3f} mm")
    print(f" RMSE cross-track   : {rmse_mm:.3f} mm")
    print(f" Max  cross-track   : {max_mm:.3f} mm")
    print(f" Overshoot          : {overshoot * 1000:.3f} mm")
    print(f" Settling (2 mm)    : {settle_s:.2f} s")
    print(f" mean |P term|      : {p_mean:.5f}")
    print(f" mean |D term|      : {d_mean:.5f}   ({ratio:.1f} % of P)")
    if latencies:
        print(f" CNN latency mean   : {lat_mean:.3f} ms")
        print(f" CNN latency p95    : {lat_p95:.3f} ms")
    print(f" CSV                : {path.name}")
    print("=" * 52)
    print("")

    log_dir = root / "evaluation_summary_final"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_log.csv"
    header = ["run_file", "method", "world", "stop_reason", "start_offset_mm",
              "samples", "run_time_s", "completion_pct", "line_visible_pct",
              "roi_mean_grey", "mae_mm", "rmse_mm", "max_abs_mm",
              "overshoot_mm", "settling_s", "mean_abs_p", "mean_abs_d",
              "d_over_p_pct", "cnn_latency_mean_ms", "cnn_latency_p95_ms"]
    new_file = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow([
            path.stem, method, world, reason, round(offset * 1000, 2),
            int(a.size), round(elapsed, 3), round(progress * 100, 3),
            round(vis_pct, 3), round(grey_mean, 3), round(mae_mm, 4),
            round(rmse_mm, 4), round(max_mm, 4), round(overshoot * 1000, 4),
            round(settle_s, 4) if settle_s == settle_s else "",
            round(p_mean, 6), round(d_mean, 6), round(ratio, 3),
            round(lat_mean, 4) if lat_mean == lat_mean else "",
            round(lat_p95, 4) if lat_p95 == lat_p95 else "",
        ])