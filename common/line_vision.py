import cv2
import numpy as np
from controller import Display

# ---------- pipeline constants (do not change independently) ----------
ROI_RATIO = 0.72          # top 72% of the camera frame
CNN_W = 160
CNN_H = 96

# ---------- detector constants (validated on prepared_v1) -------------
DARK_THRESHOLD = 90.0     # floor ~234, line ~28  -> 90 sits safely between
MIN_DARK_FRAC = 0.005     # below this -> no line in view
MAX_DARK_FRAC = 0.55      # above this -> degenerate frame, treat as no line


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def grab_bgr(camera, cam_w, cam_h):
    """Webots BGRA buffer -> OpenCV BGR array. Returns None if unavailable."""
    raw = camera.getImage()
    if raw is None:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != cam_w * cam_h * 4:
        return None
    return arr.reshape((cam_h, cam_w, 4))[:, :, :3].copy()


def roi_of(bgr):
    """Top 72% of the frame - identical to the dataset generation ROI."""
    h = bgr.shape[0]
    return bgr[0:int(h * ROI_RATIO), :, :]


def roi_gray(bgr):
    return cv2.cvtColor(roi_of(bgr), cv2.COLOR_BGR2GRAY)


def cnn_tensor(bgr):
    """ROI -> grayscale -> 160x96 -> float32 [0,1] -> (1,96,160,1)."""
    g = roi_gray(bgr)
    g = cv2.resize(g, (CNN_W, CNN_H), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32) / 255.0
    return g.reshape(1, CNN_H, CNN_W, 1)


def detect_line(gray):
    """
    Returns (error, bbox, dark_fraction).

    error : normalized [-1, +1], negative = line is LEFT, positive = RIGHT
            None if the line is genuinely not visible.
    bbox  : (x0, y0, x1, y1) in ROI pixel coordinates, or None.
    """
    mask = gray < DARK_THRESHOLD
    frac = float(mask.mean())

    if not (MIN_DARK_FRAC <= frac <= MAX_DARK_FRAC):
        return None, None, frac

    cols = mask.sum(axis=0).astype(np.float32)
    total = float(cols.sum())
    if total <= 0.0:
        return None, None, frac

    x = np.arange(gray.shape[1], dtype=np.float32)
    cx = float((x * cols).sum() / total)
    ic = (gray.shape[1] - 1) / 2.0
    error = clamp((cx - ic) / ic, -1.0, 1.0)

    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return error, bbox, frac


def error_to_box(error, roi_w, roi_h, half=34):
    """Turn a scalar predicted error into a square box (for the CNN overlay)."""
    ic = (roi_w - 1) / 2.0
    cx = int(round(ic + error * ic))
    cy = int(roi_h * 0.5)
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(roi_w - 1, cx + half)
    y1 = min(roi_h - 1, cy + half)
    return (x0, y0, x1, y1)


def render(display, bgr, roi_h, det_bbox=None, cnn_box=None,
           header="", lines=()):
    """
    Draw the annotated camera view into the Webots Display device.

    det_bbox : GREEN    - classical detector blob (reference)
    cnn_box  : RED      - where the CNN thinks the line is (hybrid only)

    IMPORTANT: the image is pasted LAST, after every annotation has been
    drawn. Pasting earlier (or returning early) sends the raw camera frame
    to the Display and every overlay silently disappears.
    """
    vis = bgr.copy()
    h, w = vis.shape[:2]
    ic = int((w - 1) / 2)

    # ROI boundary (yellow) + image centre reference (blue)
    cv2.rectangle(vis, (0, 0), (w - 1, roi_h - 1), (0, 200, 200), 1)
    cv2.line(vis, (ic, 0), (ic, roi_h - 1), (255, 0, 0), 1)

    # classical detector blob
    if det_bbox is not None:
        x0, y0, x1, y1 = det_bbox
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # CNN estimate
    if cnn_box is not None:
        x0, y0, x1, y1 = cnn_box
        cx = (x0 + x1) // 2
        cv2.line(vis, (cx, 0), (cx, roi_h - 1), (255, 0, 255), 2)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.drawMarker(vis, (cx, (y0 + y1) // 2), (0, 0, 255),
                       cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, f"CNN x={cx}", (max(2, x0), max(16, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # text overlay
    cv2.putText(vis, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 255, 255), 1)
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (6, 34 + i * 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1)

    # ---- paste LAST ----
    bgra = cv2.cvtColor(vis, cv2.COLOR_BGR2BGRA)
    if display is not None:
        ref = display.imageNew(bgra.tobytes(), Display.BGRA, w, h)
        display.imagePaste(ref, 0, 0, False)
        display.imageDelete(ref)      # MUST delete every frame or Webots leaks
    return vis