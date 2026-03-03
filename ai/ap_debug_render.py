# ============================================================
# IMPORTS
# ============================================================

import os
import cv2
import numpy as np
import json
import csv
import re
import glob
from datetime import datetime

import pydicom
from pydicom.errors import InvalidDicomError
from ultralytics import YOLO


# ============================================================
# AP Knee Overlay (Tibia + Femur SVG) — AUTO-FIT TO OUTLINES
# v2.1 — LANDMARK WIDTH FIX
#
# TIBIA: width measured in tight top band (~5%)
# FEMUR: width measured in tight bottom band (~10%)
# Clean ASCII labels
# Diagnostic width print (px + mm)
# ============================================================


# ============================================================
# OUTPUT MODE
# ============================================================

_CLEAN_MODE = True   # True = clean overlay only


# ============================================================
# YOLO CLASS IDS
# ============================================================

CLASS_FEMUR = 0
CLASS_TIBIA = 1


# ============================================================
# PATH + MODEL INITIALIZATION (FASTAPI SAFE)
# ============================================================

# Current file: Website/ai/ap_debug_render.py
AI_DIR = os.path.dirname(os.path.abspath(__file__))      # Website/ai
WEBSITE_DIR = os.path.dirname(AI_DIR)                    # Website root

# --- Model path ---
MODEL_PATH = os.path.join(WEBSITE_DIR, "AP_v5.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# --- Output directory (Website/outputs) ---
OUT_DIR = os.path.join(WEBSITE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Optional DICOM directory (if used) ---
DICOM_DIR = os.path.join(WEBSITE_DIR, "dicoms")

# --- SVG directory (inside ai folder) ---
SVG_DIR = os.path.join(AI_DIR, "implants_svg")

if not os.path.exists(SVG_DIR):
    raise FileNotFoundError(f"SVG directory not found at: {SVG_DIR}")


MAX_IMAGES_TO_PROCESS = 50

OUTLINE_THICKNESS = 6
DRAW_RAW_POLYS = False

# -----------------------------
# Tibia geometry
# -----------------------------
TIBIA_PLATEAU_BAND_FRAC = 0.07
PLATEAU_ENVELOPE_BINS = 40
PLATEAU_TRIM_FRAC = 0.20
PLATEAU_ANGLE_MAX_DEG = 3.0
TIBIA_TRAY_CLEARANCE_PX = 2
TIBIA_TOP_MARGIN_PX = 2
TIBIA_TOP_MAX_PUSH_DOWN_PX = 12

TIBIA_SIZING_BAND_TOP_FRAC = 0.00
TIBIA_SIZING_BAND_BOT_FRAC = 0.05

FEMUR_SIZING_BAND_TOP_FRAC = 0.90
FEMUR_SIZING_BAND_BOT_FRAC = 1.00

FEMUR_MIN_CONDYLE_MM = 55.0

# -----------------------------
# Femur geometry
# -----------------------------
FEMUR_ROT_MAX_DEG = 3.0
MIN_JOINT_GAP_FRAC = 0.12
FEMUR_EXTRA_GAP_PX = 18
FEMUR_TOP_MARGIN_PX = 2

# -----------------------------
# Measurement panel
# -----------------------------
MEAS_FONT_SCALE = 1.25
MEAS_THICKNESS  = 3
MEAS_PADDING    = 14
MEAS_LINE_GAP   = 12
MEAS_ALPHA      = 0.70

TIBIA_BONE_TO_IMPLANT_RATIO = 0.92
TIBIAL_SIZE_BINS = [
    ("A",  0.0,   59.25),
    ("B",  59.25, 62.30),
    ("C",  62.30, 65.40),
    ("D",  65.40, 69.00),
    ("E",  69.00, 73.05),
    ("F",  73.05, 77.05),
    ("G",  77.05, 81.00),
    ("H",  81.00, 85.55),
    ("I",  85.55, 999.0),
]

FEMUR_BONE_TO_IMPLANT_RATIO = 0.95
FEMUR_SIZE_BINS = [
    ("3",   0.0,   63.40),
    ("4",  63.40,  65.15),
    ("5",  65.15,  66.90),
    ("6",  66.90,  68.65),
    ("7",  68.65,  70.40),
    ("8",  70.40,  72.15),
    ("9",  72.15,  73.90),
    ("10", 73.90,  75.65),
    ("11", 75.65,  77.00),
    ("12", 77.00, 999.0),
]

TIBIA_SVG_FILENAME = "tibial_tray_ap.svg"
FEMUR_SVG_FILENAME = "femur_component_ap.svg"


# ============================================================
# Size mapping helpers
# ============================================================
def size_range_from_width(width_mm, bins, qc_flags):
    if (not qc_flags) or (qc_flags == "OK"):
        confidence = "high"
    elif any(k in qc_flags for k in ("missing", "fail", "no_contour", "exception")):
        confidence = "low"
    else:
        confidence = "medium"

    matching = [size for size, lo, hi in bins if lo <= width_mm <= hi]

    if not matching:
        centers = [((lo + hi) / 2.0, size) for size, lo, hi in bins]
        nearest = min(centers, key=lambda x: abs(x[0] - width_mm))[1]
        idx = [s for s, _, _ in bins].index(nearest)
        lo_idx = max(0, idx - 1)
        hi_idx = min(len(bins) - 1, idx + 1)
        return bins[lo_idx][0], bins[hi_idx][0], "low"

    indices = [i for i, (s, _, _) in enumerate(bins) if s in matching]
    lo_idx = max(0, min(indices) - 1)
    hi_idx = min(len(bins) - 1, max(indices) + 1)
    return bins[lo_idx][0], bins[hi_idx][0], confidence


# ============================================================
# YOLO / geometry helpers
# ============================================================
def load_yolo_seg_polygons(txt_path, img_w, img_h):
    polys = []
    if not os.path.exists(txt_path):
        return polys
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            polys.append((cls, pts.astype(np.int32)))
    return polys


def polygon_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def outline_width_from_contour(cnt, trim_frac=0.03, min_frac_of_bbox=0.90):
    xs = cnt[:, 0, 0].astype(np.float32)
    xL = float(np.percentile(xs, 100.0 * trim_frac))
    xR = float(np.percentile(xs, 100.0 * (1.0 - trim_frac)))
    w_robust = max(1.0, xR - xL)
    w_bbox = max(1.0, float(xs.max() - xs.min()))
    w_clamped = w_robust
    if w_robust < float(min_frac_of_bbox) * w_bbox:
        w_clamped = float(min_frac_of_bbox) * w_bbox
        cx = 0.5 * (float(xs.max()) + float(xs.min()))
        half = 0.5 * w_clamped
        xL = cx - half
        xR = cx + half
    return xL, xR, w_clamped


def width_from_contour_y_band(cnt, y0, y1, trim_frac=0.06):
    pts = cnt[:, 0, :].astype(np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    m = (ys >= float(y0)) & (ys <= float(y1))
    xs_band = xs[m] if np.count_nonzero(m) >= 30 else xs
    xL = float(np.percentile(xs_band, 100.0 * trim_frac))
    xR = float(np.percentile(xs_band, 100.0 * (1.0 - trim_frac)))
    w = max(1.0, xR - xL)
    return xL, xR, w


def clamp_deg(x, lim):
    return max(-lim, min(lim, float(x)))


def clamp_tray_not_too_low(tib_cnt, tray_pts_px, max_drop_frac=0.08):
    ys = tib_cnt[:, 0, 1].astype(np.float32)
    y_min = float(ys.min())
    y_max = float(ys.max())
    h = max(1.0, y_max - y_min)
    tray_top_y = float(np.min(tray_pts_px[:, 1]))
    allowed_top_y = y_min + float(max_drop_frac) * h
    if tray_top_y > allowed_top_y:
        dy = allowed_top_y - tray_top_y
        tray_pts_px[:, 1] += dy
        return tray_pts_px, dy
    return tray_pts_px, 0.0


def find_dicom_for_uid(uid):
    exact = os.path.join(DICOM_DIR, uid + ".dcm")
    if os.path.exists(exact):
        return exact
    matches = glob.glob(os.path.join(DICOM_DIR, uid + "*.dcm"))
    return matches[0] if matches else None


def load_pixel_spacing_mm(dicom_path, png_w=None, png_h=None):
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None or len(ps) < 2:
        ps = getattr(ds, "ImagerPixelSpacing", None)
    if ps is None or len(ps) < 2:
        raise RuntimeError("DICOM missing PixelSpacing/ImagerPixelSpacing")
    dicom_y_mm = float(ps[0])
    dicom_x_mm = float(ps[1])
    dicom_cols = float(getattr(ds, "Columns", 0) or 0)
    dicom_rows = float(getattr(ds, "Rows", 0) or 0)
    if png_w and png_w > 0 and dicom_cols > 0:
        dicom_x_mm *= (dicom_cols / float(png_w))
    if png_h and png_h > 0 and dicom_rows > 0:
        dicom_y_mm *= (dicom_rows / float(png_h))
    return dicom_x_mm, dicom_y_mm


def clamp_femur_not_below_joint(fem_pts_px, joint_y, margin_px=2):
    bottom = float(np.max(fem_pts_px[:, 1]))
    allowed_bottom = float(joint_y) - float(margin_px)
    if bottom > allowed_bottom:
        dy = allowed_bottom - bottom
        fem_pts_px[:, 1] += dy
        return fem_pts_px, dy
    return fem_pts_px, 0.0


# ============================================================
# Landmark-based width measurement
# ============================================================
def tibia_plateau_ml_width(tib_cnt, band_top_frac=TIBIA_SIZING_BAND_TOP_FRAC,
                            band_bot_frac=TIBIA_SIZING_BAND_BOT_FRAC,
                            trim_frac=0.04):
    pts = tib_cnt[:, 0, :].astype(np.float32)
    ys = pts[:, 1]
    y_min = float(ys.min())
    y_max = float(ys.max())
    h = max(1.0, y_max - y_min)
    y0 = y_min + band_top_frac * h
    y1 = y_min + band_bot_frac * h
    if (y1 - y0) < 0.03 * h:
        y1 = y_min + 0.08 * h
    return width_from_contour_y_band(tib_cnt, y0, y1, trim_frac=trim_frac)


def femur_condylar_ml_width(fem_cnt, band_top_frac=FEMUR_SIZING_BAND_TOP_FRAC,
                             band_bot_frac=FEMUR_SIZING_BAND_BOT_FRAC,
                             trim_frac=0.04):
    pts = fem_cnt[:, 0, :].astype(np.float32)
    ys = pts[:, 1]
    y_min = float(ys.min())
    y_max = float(ys.max())
    h = max(1.0, y_max - y_min)
    y0 = y_min + band_top_frac * h
    y1 = y_min + band_bot_frac * h
    if (y1 - y0) < 0.03 * h:
        y0 = y_max - 0.12 * h
    return width_from_contour_y_band(fem_cnt, y0, y1, trim_frac=trim_frac)


# ============================================================
# Plateau segment (for tibia angle + seat y)
# ============================================================
def plateau_segment_from_tibia_contour(tib_cnt, band_frac=TIBIA_PLATEAU_BAND_FRAC):
    pts = tib_cnt[:, 0, :].astype(np.float32)
    xs_all = pts[:, 0]
    ys_all = pts[:, 1]
    y_min = float(np.min(ys_all))
    y_max = float(np.max(ys_all))
    cutoff = y_min + float(band_frac) * (y_max - y_min)
    band_mask = ys_all <= cutoff
    band = pts[band_mask]
    if band.shape[0] < 20:
        band = pts

    x_min = float(np.min(band[:, 0]))
    x_max = float(np.max(band[:, 0]))
    edges = np.linspace(x_min, x_max, int(PLATEAU_ENVELOPE_BINS) + 1)
    env = []
    for i in range(len(edges) - 1):
        a = edges[i]; b = edges[i + 1]
        m = (band[:, 0] >= a) & (band[:, 0] < b)
        if not np.any(m):
            continue
        chunk = band[m]
        y_target = float(np.percentile(chunk[:, 1], 10))
        j = int(np.argmin(np.abs(chunk[:, 1] - y_target)))
        env.append([float(np.mean(chunk[:, 0])), float(chunk[j, 1])])

    if len(env) < 10:
        env = band.tolist()
    env = np.array(env, dtype=np.float32)
    env = env[np.argsort(env[:, 0])]

    n = env.shape[0]
    k = int(PLATEAU_TRIM_FRAC * n) if n >= 10 else 0
    env_fit = env[k:n - k] if (n - 2 * k) >= 6 else env

    vx, vy, x0, y0 = cv2.fitLine(env_fit, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if vx < 0:
        vx, vy = -vx, -vy

    ang_deg = float(np.degrees(np.arctan2(vy, vx)))
    if abs(ang_deg) > float(PLATEAU_ANGLE_MAX_DEG):
        ang_deg = float(np.sign(ang_deg) * float(PLATEAU_ANGLE_MAX_DEG))
        rad = np.deg2rad(ang_deg)
        vx = float(np.cos(rad))
        vy = float(np.sin(rad))

    t = (env_fit[:, 0] - x0) * vx + (env_fit[:, 1] - y0) * vy
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    pl = (int(x0 + tmin * vx), int(y0 + tmin * vy))
    pr = (int(x0 + tmax * vx), int(y0 + tmax * vy))
    t_mid = 0.5 * (tmin + tmax)
    plateau_y = float(y0 + t_mid * vy)
    return pl, pr, ang_deg, plateau_y


# ============================================================
# Tibia top envelope (safety clamp)
# ============================================================
def tibia_top_envelope_y(cnt, top_frac=0.18, n_bins=180, trim_frac=0.12):
    pts = cnt[:, 0, :].astype(np.float32)
    xs = pts[:, 0]; ys = pts[:, 1]
    x_min = float(xs.min()); x_max = float(xs.max())
    y_min = float(ys.min()); y_max = float(ys.max())
    h = y_max - y_min
    if x_max - x_min < 5 or h < 5:
        return None, None

    y_cut = y_min + float(top_frac) * h
    m_top = ys <= y_cut
    top = pts[m_top] if m_top.sum() >= 20 else pts

    x_min2 = float(np.min(top[:, 0])); x_max2 = float(np.max(top[:, 0]))
    xL = x_min2 + float(trim_frac) * (x_max2 - x_min2)
    xR = x_max2 - float(trim_frac) * (x_max2 - x_min2)
    m_mid = (top[:, 0] >= xL) & (top[:, 0] <= xR)
    top = top[m_mid] if np.any(m_mid) else top

    x_min3 = float(np.min(top[:, 0])); x_max3 = float(np.max(top[:, 0]))
    edges = np.linspace(x_min3, x_max3, int(n_bins) + 1)
    xs_c, ys_t = [], []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        m = (top[:, 0] >= a) & (top[:, 0] < b)
        if not np.any(m):
            continue
        chunk_y = top[m][:, 1]
        xs_c.append(0.5 * (a + b))
        ys_t.append(float(np.percentile(chunk_y, 10)))

    if len(xs_c) < 10:
        return None, None
    return np.array(xs_c, dtype=np.float32), np.array(ys_t, dtype=np.float32)


def clamp_points_below_tibia_top(tib_cnt, pts_px,
                                  margin_px=TIBIA_TOP_MARGIN_PX,
                                  max_push_down_px=TIBIA_TOP_MAX_PUSH_DOWN_PX):
    xs_env, ys_env = tibia_top_envelope_y(tib_cnt, top_frac=0.18, n_bins=180, trim_frac=0.12)
    if xs_env is None:
        return pts_px, 0.0
    xs = pts_px[:, 0].astype(np.float32)
    ys = pts_px[:, 1].astype(np.float32)
    x_min = float(xs_env.min()); x_max = float(xs_env.max())
    xs_clip = np.clip(xs, x_min, x_max)
    y_allowed = np.interp(xs_clip, xs_env, ys_env) + float(margin_px)
    needed = float(np.max(y_allowed - ys))
    if needed > 0:
        needed = min(needed, float(max_push_down_px))
        pts_px[:, 1] += needed
        return pts_px, needed
    return pts_px, 0.0


# ============================================================
# SVG helpers
# ============================================================
def parse_viewbox(svg_text):
    vb = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_text, re.IGNORECASE)
    if vb:
        nums = list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", vb.group(1))))
        if len(nums) == 4:
            return nums[0], nums[1], max(1e-6, nums[2]), max(1e-6, nums[3])
    return 0.0, 0.0, 1.0, 1.0


def sample_cubic(p0, p1, p2, p3, n=20):
    ts = np.linspace(0, 1, n, dtype=np.float32)
    out = []
    for t in ts:
        a = (1 - t)
        p = (a*a*a)*p0 + 3*(a*a)*t*p1 + 3*a*(t*t)*p2 + (t*t*t)*p3
        out.append([float(p[0]), float(p[1])])
    return out


def path_to_points(d):
    tokens = re.findall(r"[MLCmlcZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    pts = []
    i = 0; cmd = None
    cur = np.array([0.0, 0.0], dtype=np.float32)
    start = None

    def next_xy():
        nonlocal i
        x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
        return np.array([x, y], dtype=np.float32)

    while i < len(tokens):
        t = tokens[i]
        if re.match(r"[MLCmlcZz]", t):
            cmd = t; i += 1
            if cmd in "Zz" and start is not None:
                pts.append([float(start[0]), float(start[1])])
                cur = start.copy()
            continue

        if cmd in ("M", "L"):
            p = next_xy(); cur = p
            if start is None: start = cur.copy()
            pts.append([float(cur[0]), float(cur[1])])
        elif cmd in ("m", "l"):
            p = next_xy(); cur = cur + p
            if start is None: start = cur.copy()
            pts.append([float(cur[0]), float(cur[1])])
        elif cmd == "C":
            p1 = next_xy(); p2 = next_xy(); p3 = next_xy()
            pts.extend(sample_cubic(cur, p1, p2, p3, n=20)); cur = p3
        elif cmd == "c":
            p1 = cur + next_xy(); p2 = cur + next_xy(); p3 = cur + next_xy()
            pts.extend(sample_cubic(cur, p1, p2, p3, n=20)); cur = p3
        else:
            i += 1

    return np.array(pts, dtype=np.float32) if len(pts) >= 3 else None


def load_svg_points_normalized(svg_path):
    with open(svg_path, "r", encoding="utf-8") as f:
        s = f.read()
    vb_x, vb_y, vb_w, vb_h = parse_viewbox(s)
    all_pts = []
    for m in re.finditer(r"<polygon[^>]*points\s*=\s*\"([^\"]+)\"", s, re.IGNORECASE):
        pts_str = m.group(1).strip()
        pts = []
        for t in re.split(r"\s+", pts_str):
            t = t.strip().strip(",")
            if not t: continue
            x_str, y_str = t.split(",")
            pts.append([float(x_str), float(y_str)])
        if len(pts) >= 3:
            all_pts.append(np.array(pts, dtype=np.float32))
    for m in re.finditer(r"<path[^>]*d\s*=\s*\"([^\"]+)\"", s, re.IGNORECASE):
        pts = path_to_points(m.group(1))
        if pts is not None and pts.shape[0] >= 3:
            all_pts.append(pts)
    if not all_pts:
        raise RuntimeError(f"No <polygon> or <path> found in {svg_path}")
    pts = np.vstack(all_pts).astype(np.float32)
    pts[:, 0] = (pts[:, 0] - vb_x) / vb_w
    pts[:, 1] = (pts[:, 1] - vb_y) / vb_h
    pts[:, 0] = np.clip(pts[:, 0], -2.0, 3.0)
    pts[:, 1] = np.clip(pts[:, 1], -2.0, 3.0)
    return pts


def find_svg_path(filename):
    """
    Find implant asset inside ai/implants_svg.
    Works on Windows and Linux (Render).
    """

    # Direct match
    exact_path = os.path.join(SVG_DIR, filename)
    if os.path.exists(exact_path):
        return exact_path

    # Fallback: match by base name (handles .svg/.json flexibility)
    base = os.path.splitext(filename)[0]
    matches = glob.glob(os.path.join(SVG_DIR, base + ".*"))

    if matches:
        # Prefer .svg first
        matches_sorted = sorted(
            matches,
            key=lambda p: (not p.lower().endswith(".svg"), p.lower())
        )
        return matches_sorted[0]

    raise FileNotFoundError(
        f"{filename} not found in {SVG_DIR}"
    )

# ============================================================
# Point transforms + auto-fit
# ============================================================
def transform_norm_points_to_pixels(pts_norm, center_px, target_width_px, angle_deg):
    pts = pts_norm.copy().astype(np.float32)
    pts[:, 0] -= 0.5; pts[:, 1] -= 0.5
    pts *= float(target_width_px)
    theta = np.deg2rad(float(angle_deg))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    pts = pts @ R.T
    pts[:, 0] += float(center_px[0]); pts[:, 1] += float(center_px[1])
    return pts


def rescale_points_about_center(pts_px, center_px, scale):
    c = np.array([float(center_px[0]), float(center_px[1])], dtype=np.float32)
    return c + float(scale) * (pts_px - c)


def auto_fit_svg_to_outline_width(pts_px, seat_px, outline_xL, outline_xR, center_to_outline=True):
    desired_w = float(outline_xR - outline_xL)
    cur_w = float(np.max(pts_px[:, 0]) - np.min(pts_px[:, 0]))
    if cur_w > 1e-6 and desired_w > 1.0:
        s = desired_w / cur_w
        pts_px = rescale_points_about_center(pts_px, seat_px, s)
    if center_to_outline:
        svg_cx = 0.5 * (float(np.max(pts_px[:, 0])) + float(np.min(pts_px[:, 0])))
        target_cx = 0.5 * (float(outline_xL) + float(outline_xR))
        dx = target_cx - svg_cx
        pts_px[:, 0] += dx
        seat_px = (int(seat_px[0] + dx), int(seat_px[1]))
    return pts_px, seat_px


# ============================================================
# Drawing + measurement panel
# ============================================================
def draw_polygon_outline(img, pts_px, color, thickness):
    poly = np.int32(pts_px).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness)


def draw_filled_polygon(img, pts_px, color, alpha=0.22):
    """Semi-transparent fill for clean mode."""
    overlay = img.copy()
    poly = np.int32(pts_px).reshape(-1, 1, 2)
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_measurement_panel_top_only(img, lines):
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = MEAS_FONT_SCALE; thick = MEAS_THICKNESS
    pad = MEAS_PADDING; gap = MEAS_LINE_GAP
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    max_w = max(w for (w, h) in sizes)
    line_h = max(h for (w, h) in sizes)
    block_w = max_w + 2 * pad
    block_h = len(lines) * (line_h + gap) + 2 * pad - gap
    H, W = img.shape[:2]
    x0, y0 = 20, 20
    x1 = min(W - 1, x0 + block_w); y1 = min(H - 1, y0 + block_h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, MEAS_ALPHA, img, 1 - MEAS_ALPHA, 0, img)
    y = y0 + pad + line_h
    for t in lines:
        cv2.putText(img, t, (x0 + pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h + gap


# ============================================================
# Logging
# ============================================================
def append_transform_records(out_dir, record):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "overlay_transforms_fit.csv")
    csv_exists = os.path.exists(csv_path)
    fieldnames = list(record.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            w.writeheader()
        w.writerow(record)
    jsonl_path = os.path.join(out_dir, "overlay_transforms_fit.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================
# Core renderer
# ============================================================
def render_one(image_path, label_path=None, out_path=None, debug=True):
    # -------------------------------------------------------
    # Load image safely
    # -------------------------------------------------------
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    qc = []
    uid = os.path.splitext(os.path.basename(image_path))[0]

    # -------------------------------------------------------
    # Load segmentation polygons
    # -------------------------------------------------------
    polys = []

    try:
        results = model(image_path)[0]

        if results.masks is None:
            qc.append("no_masks_detected")
        else:
            for seg, cls in zip(results.masks.xy, results.boxes.cls):
                pts = np.array(seg, dtype=np.int32)
                polys.append((int(cls), pts))

    except Exception as e:
        qc.append("yolo_inference_fail:" + str(e).replace("\n", " "))

    # -------------------------------------------------------
    # Load DICOM pixel spacing
    # -------------------------------------------------------
    x_mm_per_px = None
    dicom_path = None

    try:
        dicom_path = find_dicom_for_uid(uid)
    except Exception:
        dicom_path = None

    if dicom_path:
        try:
            x_mm_per_px, _ = load_pixel_spacing_mm(
                dicom_path,
                png_w=w,
                png_h=h
            )
        except Exception as e:
            qc.append("dicom_spacing_fail:" + str(e).replace("\n", " "))
    else:
        qc.append("missing_dicom")

    # -------------------------------------------------------
    # Initialize record
    # -------------------------------------------------------
    record = {
        "uid": uid,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "qc_flags": "",
        "tibia_size_min": "",
        "tibia_size_max": "",
        "tibia_size_confidence": "",
        "femur_size_min": "",
        "femur_size_max": "",
        "femur_size_confidence": "",
        "tibia_ml_px": "",
        "tibia_ml_mm": "",
        "femur_ml_px": "",
        "femur_ml_mm": "",
        "x_mm_per_px": x_mm_per_px if x_mm_per_px else "",
    }

    # -------------------------------------------------------
    # Extract contours
    # -------------------------------------------------------
    tib_cnt = None
    fem_cnt = None

    tibia_polys = [p for (cls, p) in polys if cls == CLASS_TIBIA]
    femur_polys = [p for (cls, p) in polys if cls == CLASS_FEMUR]

    if tibia_polys:
        tib = max(tibia_polys, key=lambda p: cv2.contourArea(p))
        tib_cnt = largest_contour(polygon_to_mask(tib, h, w))
        if tib_cnt is None:
            qc.append("tibia_no_contour")
    else:
        qc.append("missing_tibia")

    if femur_polys:
        fem = max(femur_polys, key=lambda p: cv2.contourArea(p))
        fem_cnt = largest_contour(polygon_to_mask(fem, h, w))
        if fem_cnt is None:
            qc.append("femur_no_contour")
    else:
        qc.append("missing_femur")

    # -------------------------------------------------------
    # Auto swap
    # -------------------------------------------------------
    if tib_cnt is not None and fem_cnt is not None:
        tib_c = contour_centroid(tib_cnt)
        fem_c = contour_centroid(fem_cnt)
        if tib_c and fem_c and tib_c[1] < fem_c[1]:
            tib_cnt, fem_cnt = fem_cnt, tib_cnt
            qc.append("warn:auto_swapped_femur_tibia")

    # -------------------------------------------------------
    # Debug outlines
    # -------------------------------------------------------
    if debug:
        if DRAW_RAW_POLYS:
            for cls, pts in polys:
                cv2.polylines(img, [pts], True, (0, 255, 255), 2)

        if tib_cnt is not None:
            cv2.drawContours(img, [tib_cnt], -1, (0, 255, 0), 3)

        if fem_cnt is not None:
            cv2.drawContours(img, [fem_cnt], -1, (0, 165, 255), 3)

    # -------------------------------------------------------
    # TIBIA
    # -------------------------------------------------------
    tib_ok = False
    tib_plateau_ang = 0.0
    tib_plateau_y = 0.0
    tib_outline_w = 0.0

    if tib_cnt is not None:
        tib_c = contour_centroid(tib_cnt)

        if tib_c is None:
            qc.append("tibia_centroid_fail")
        else:
            txL_fit, txR_fit, tib_sizing_w = tibia_plateau_ml_width(tib_cnt)

            tib_ys = tib_cnt[:, 0, 1].astype(np.float32)
            tib_ymin = float(np.min(tib_ys))
            tib_ymax = float(np.max(tib_ys))
            tib_h = max(1.0, tib_ymax - tib_ymin)

            txL_fit, txR_fit, tib_outline_w = width_from_contour_y_band(
                tib_cnt,
                tib_ymin,
                tib_ymin + 0.20 * tib_h,
                trim_frac=0.06
            )

            tib_ml_mm = None
            if x_mm_per_px:
                tib_ml_mm = tib_sizing_w * x_mm_per_px * TIBIA_BONE_TO_IMPLANT_RATIO

            record["tibia_ml_px"] = round(tib_sizing_w, 1)
            record["tibia_ml_mm"] = round(tib_ml_mm, 1) if tib_ml_mm else ""

            pl, pr, tib_plateau_ang, tib_plateau_y = plateau_segment_from_tibia_contour(
                tib_cnt,
                TIBIA_PLATEAU_BAND_FRAC
            )

            if debug:
                cv2.line(img, pl, pr, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(img, pl, pr, (0, 255, 255), 2, cv2.LINE_AA)

            qc_flags_str = "|".join(qc) if qc else "OK"

            if tib_ml_mm:
                tmin, tmax, tconf = size_range_from_width(
                    tib_ml_mm,
                    TIBIAL_SIZE_BINS,
                    qc_flags_str
                )
            else:
                tmin, tmax, tconf = "", "", "low"

            record["tibia_size_min"] = tmin
            record["tibia_size_max"] = tmax
            record["tibia_size_confidence"] = tconf

            tib_outline_cx = 0.5 * (txL_fit + txR_fit)
            tib_seat = (int(tib_outline_cx), int(tib_plateau_y))

            try:
                tib_svg_path = find_svg_path(TIBIA_SVG_FILENAME)
                tib_pts_norm = load_svg_points_normalized(tib_svg_path)

                tib_pts_px = transform_norm_points_to_pixels(
                    tib_pts_norm,
                    tib_seat,
                    tib_outline_w,
                    tib_plateau_ang
                )

                tib_pts_px, tib_seat = auto_fit_svg_to_outline_width(
                    tib_pts_px,
                    tib_seat,
                    txL_fit,
                    txR_fit
                )

                tib_pts_px, _ = clamp_points_below_tibia_top(tib_cnt, tib_pts_px)
                tib_pts_px, _ = clamp_tray_not_too_low(tib_cnt, tib_pts_px)

                if not debug:
                    draw_filled_polygon(img, tib_pts_px, (255, 100, 60), 0.22)

                draw_polygon_outline(img, tib_pts_px, (0, 0, 255), OUTLINE_THICKNESS)

                tib_ok = True

            except Exception as e:
                qc.append("tibia_svg_fail:" + str(e).replace("\n", " "))

    # -------------------------------------------------------
    # FEMUR
    # -------------------------------------------------------
    if fem_cnt is not None and tib_ok:

        fxL_fit, fxR_fit, fem_sizing_w = femur_condylar_ml_width(fem_cnt)

        fem_ml_mm = None
        if x_mm_per_px:
            fem_ml_mm = fem_sizing_w * x_mm_per_px * FEMUR_BONE_TO_IMPLANT_RATIO

        record["femur_ml_px"] = round(fem_sizing_w, 1)
        record["femur_ml_mm"] = round(fem_ml_mm, 1) if fem_ml_mm else ""

        fem_outline_cx = 0.5 * (fxL_fit + fxR_fit)
        fem_angle = clamp_deg(tib_plateau_ang, FEMUR_ROT_MAX_DEG)
        fem_seat = (int(fem_outline_cx), int(np.max(fem_cnt[:, 0, 1])))

        qc_flags_str = "|".join(qc) if qc else "OK"

        if fem_ml_mm:
            fmin, fmax, fconf = size_range_from_width(
                fem_ml_mm,
                FEMUR_SIZE_BINS,
                qc_flags_str
            )
        else:
            fmin, fmax, fconf = "", "", "low"

        record["femur_size_min"] = fmin
        record["femur_size_max"] = fmax
        record["femur_size_confidence"] = fconf

        try:
            fem_svg_path = find_svg_path(FEMUR_SVG_FILENAME)
            fem_pts_norm = load_svg_points_normalized(fem_svg_path)

            fem_pts_px = transform_norm_points_to_pixels(
                fem_pts_norm,
                fem_seat,
                float(fxR_fit - fxL_fit),
                fem_angle
            )

            fem_pts_px, fem_seat = auto_fit_svg_to_outline_width(
                fem_pts_px,
                fem_seat,
                fxL_fit,
                fxR_fit
            )

            if not debug:
                draw_filled_polygon(img, fem_pts_px, (255, 0, 255), 0.22)

            draw_polygon_outline(img, fem_pts_px, (255, 0, 255), OUTLINE_THICKNESS)

        except Exception as e:
            qc.append("femur_svg_fail:" + str(e).replace("\n", " "))

    # -------------------------------------------------------
    # FINAL PANEL + SAVE
    # -------------------------------------------------------
    record["qc_flags"] = "|".join(qc) if qc else "OK"

    lines = [
        f"TIBIA SIZE: {record['tibia_size_min']}-{record['tibia_size_max']} "
        f"({record['tibia_size_confidence']})"
        if record["tibia_size_min"] else "TIBIA SIZE: --",

        f"FEMUR SIZE: {record['femur_size_min']}-{record['femur_size_max']} "
        f"({record['femur_size_confidence']})"
        if record["femur_size_min"] else "FEMUR SIZE: --",
    ]

    draw_measurement_panel_top_only(img, lines)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)

    return record