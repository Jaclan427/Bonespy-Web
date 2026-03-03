"""
lat_debug_render_v4_2.py
========================
Persona lateral knee overlay — v4.2

CHANGES FROM v4.1:
  [1] TIBIA FIX [CRITICAL]: Replaced PCA with top-8% polyfit for plateau slope.
      PCA measured the tibia shaft (~90 deg), always clamped to +/-20, causing
      massive tibRot errors (+38 deg for flat plateaus). polyfit on the top 8%
      of the contour correctly reads the resected surface. Clamp stays at +/-20.

  [2] FEMUR FIX: fem_bot_y now uses 95th-percentile of femur ys (not bbox max).
      bbox max included noisy joint-line pixels; 95th pct is a stable condyle anchor.

  [3] FEMUR ROTATION: fem_rot_deg now computed from bottom-15% condyle edge slope
      (clamped +/-12 deg) instead of hardcoded 0. Aligns implant to actual
      condyle orientation in the image.

  [4] MODEL: LateralV4.pt — now detects patella as a real segmentation class.
      CLASS_PATELLA = 2  (verify with `print(model.names)` and adjust if different)

  [2] PATELLA CLASS → antFlip (replaces image-pixel heuristic):
      - Uses YOLO patella polygon centroid vs femur polygon centroid
      - patella.cx < femur.cx  →  patSide = +1  →  leg_mirror = False (right knee standard)
      - patella.cx > femur.cx  →  patSide = -1  →  leg_mirror = True  (left/mirrored knee)
      - Falls back to old pixel heuristic only if patella not detected

  [3] TIBIA ROTATION — PCA-based, always enabled:
      TIB_TEMPLATE_ANGLE_DEG = -18.23  (measured from insert line in normalized coords)
      INSERT_ANCHOR_N = [0.5102, 0.2742]  (insert midpoint in normalized coords)

      Rotation formula (DIFFERENT for mirror vs non-mirror — derived from template geometry):
        Non-mirror:  tib_rot = slope_deg - TIB_TEMPLATE_ANGLE_DEG  =  slope_deg + 18.23
        Mirror:      tib_rot = slope_deg + TIB_TEMPLATE_ANGLE_DEG  =  slope_deg - 18.23
      This cancels the built-in -18.23° slope in the frozen tibia geometry and applies
      the real detected plateau angle instead.

  [4] ROTATION ANCHOR: insert midpoint (INSERT_ANCHOR_N), not tib_anchor_norm.
      This keeps the bone-implant interface stable during rotation.

  [5] femur_lat.json: must include "anterior_hint_norm" key.
      Quick fix: open femur_lat.json and add  "anterior_hint_norm": [0.5, 0.1]
      (adjusting to wherever the crescent/anterior edge of your femur template is).

  [6] HUD updated:  mir=  antFlip=  patSide=  slope=  tibRot=  all shown

GEOMETRY CONSTANTS (frozen — only change if you edit tibia_lat.json):
  TIB_TEMPLATE_ANGLE_DEG = -18.23
    source: insert p1_n=(0.0204,0.4355) p2_n=(1.0,0.1129) → arctan2(-0.3226,0.9796)
  INSERT_ANCHOR_N = [0.5102, 0.2742]
    source: insert midpoint = ((68+116)/2−67)/49, ((141+121)/2−114)/62

RUN:
  python lat_debug_render_v4_2.py                 # default debug, 20 images
  python lat_debug_render_v4_2.py --clean          # clean output, no HUD
  python lat_debug_render_v4_2.py --max_images 5  # quick test
"""

from __future__ import annotations

import sys
import os
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ============================================================
# SET THIS TO True FOR CLEAN OUTPUT (no YOLO masks/lines)
# SET THIS TO False FOR DEBUG OUTPUT (all overlays visible)
# ============================================================
_CLEAN_MODE = True   # <-- change to True for clean run
# ============================================================

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CLASS_FEMUR   = 0   # 'Distal Femur'
CLASS_PATELLA = 1   # 'Patella'
CLASS_TIBIA   = 2   # 'Proximal Tibia'

DEFAULT_MM_PER_PX = 0.143
SCALE_FRAC     = 0.95   # femur template scale fraction
TIB_SCALE_FRAC = 0.88   # tibia template scale fraction (narrower than fem ratio alone)
TIB_FROM_FEM_RATIO = 49.0 / 64.0

# Tibia rotation constants — derived from v3.1 frozen tibia_lat.json geometry.
# DO NOT CHANGE unless you replace the tibia_lat.json with new geometry.
#
# Insert line in normalized [0,1] space:
#   p1_n = ((68−67)/49, (141−114)/62) = (0.0204, 0.4355)
#   p2_n = ((116−67)/49, (121−114)/62) = (1.0,   0.1129)
#   angle = arctan2(0.1129−0.4355, 1.0−0.0204) = arctan2(−0.3226, 0.9796) = −18.23°
TIB_TEMPLATE_ANGLE_DEG = -18.23

# Insert midpoint in normalized space:
#   x = (92−67)/49 = 0.5102,  y = (131−114)/62 = 0.2742
INSERT_ANCHOR_N = np.array([0.5102, 0.2742], dtype=np.float32)

# Patella heuristic fallback (used only if YOLO patella class not detected)
PATELLA_THRESH_PCT    = 0.90
PATELLA_MIN_AREA_FRAC = 0.02
PATELLA_ROI_X_PAD_FRAC  = 0.80
PATELLA_ROI_Y_FRAC_TOP  = 0.05
PATELLA_ROI_Y_FRAC_BOT  = 0.95


# ---------------------------------------------------------------------------
# PERSONA SIZING
# ---------------------------------------------------------------------------
FEMUR_NARROW_AP = {
    1:48.1, 2:50.7, 3:51.9, 4:54.0, 5:56.0,
    6:59.0, 7:60.1, 8:62.1, 9:64.6, 10:66.6, 11:69.3
}
FEMUR_STANDARD_AP = {
    3:53.2, 4:55.6, 5:57.2, 6:59.6, 7:62.1,
    8:63.8, 9:66.2, 10:68.5, 11:71.1, 12:75.2
}
TIBIA_LATERAL_AP = {
    "A":35.1, "B":37.2, "C":39.5, "D":41.8, "E":44.6,
    "F":47.4, "G":50.2, "H":53.3, "I":56.7
}
FEM_AP_MIN, FEM_AP_MAX = 48.1, 75.2
TIB_AP_MIN, TIB_AP_MAX = 35.1, 56.7


def snap_femur(ap_mm):
    if ap_mm < FEM_AP_MIN or ap_mm > FEM_AP_MAX:
        return "OOR", ap_mm, None, False
    best_l, best_d = "OOR", float("inf")
    for sz, ap in FEMUR_NARROW_AP.items():
        d = abs(ap - ap_mm)
        if d < best_d: best_d = d; best_l = f"{sz}N"
    for sz, ap in FEMUR_STANDARD_AP.items():
        d = abs(ap - ap_mm)
        if d < best_d: best_d = d; best_l = f"{sz}S"
    return best_l, ap_mm, best_d, True


def snap_tibia(lat_ap_mm):
    if lat_ap_mm < TIB_AP_MIN or lat_ap_mm > TIB_AP_MAX:
        return "OOR", lat_ap_mm, None, False
    best_l, best_d = "OOR", float("inf")
    for sz, ap in TIBIA_LATERAL_AP.items():
        d = abs(ap - lat_ap_mm)
        if d < best_d: best_d = d; best_l = sz
    return best_l, lat_ap_mm, best_d, True


# ---------------------------------------------------------------------------
# SVG/JSON LOADER
# ---------------------------------------------------------------------------
def load_svg_json(svg_dir, name):
    path = os.path.join(svg_dir, name + ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            out[k] = np.array(v, dtype=np.float32)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# PIXEL SPACING
# ---------------------------------------------------------------------------
def extract_pixel_spacing_batch(dicom_dir, png_dir, silent=False):
    try:
        import pydicom
    except ImportError:
        if not silent: print("WARNING: pip install pydicom")
        return 0
    if not os.path.isdir(dicom_dir):
        if not silent: print(f"WARNING: DICOM dir not found: {dicom_dir}")
        return 0
    png_uids = {os.path.splitext(f)[0] for f in os.listdir(png_dir)
                if os.path.isfile(os.path.join(png_dir, f))}
    written = skipped = 0
    for fname in os.listdir(dicom_dir):
        uid = os.path.splitext(fname)[0]
        if uid not in png_uids: continue
        json_path = os.path.join(png_dir, uid + ".json")
        if os.path.exists(json_path): skipped += 1; continue
        try:
            ds = pydicom.dcmread(os.path.join(dicom_dir, fname), stop_before_pixels=True)
            ps = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
            if ps is None: continue
            v0 = float(ps[0]); v1 = float(ps[1]) if len(ps) > 1 else v0
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"PixelSpacing": v0, "PixelSpacing_raw": [v0, v1], "source_dicom": fname}, f)
            written += 1
        except Exception: continue
    if not silent: print(f"PixelSpacing: {written} written, {skipped} skipped")
    return written


def get_pixel_spacing(image_path, default_mm_per_px):
    json_path = os.path.splitext(image_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            ps = d.get("PixelSpacing") or d.get("pixel_spacing")
            if ps is not None:
                val = float(ps[0]) if isinstance(ps, list) else float(ps)
                if 0.05 < val < 1.0:
                    return val, "DICOM"
        except Exception: pass
    return float(default_mm_per_px), "DEFAULT"


# ---------------------------------------------------------------------------
# YOLO POLYGON IO
# ---------------------------------------------------------------------------
def read_polys(txt_path, img_w, img_h):
    polys = []
    if not os.path.exists(txt_path): return polys
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 3: continue
            cls = int(float(p[0]))
            pts = np.array(list(map(float, p[1:])), dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= img_w; pts[:, 1] *= img_h
            polys.append((cls, pts.astype(np.int32)))
    return polys


def biggest_poly(polys):
    if not polys: return None
    return polys[int(np.argmax([cv2.contourArea(p) for p in polys]))]


def poly_bbox(poly):
    xs = poly[:, 0].astype(np.int32); ys = poly[:, 1].astype(np.int32)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def poly_centroid(poly):
    M = cv2.moments(poly)
    if M["m00"] == 0: return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def femur_condyle_width(poly):
    """Bottom 60% of femur polygon width — condyle only, skips shaft flare."""
    ys = poly[:, 1].astype(np.float32)
    thr = ys.min() + 0.40 * (ys.max() - ys.min())
    xs = poly[:, 0][ys >= thr].astype(np.float32)
    if xs.size == 0: xs = poly[:, 0].astype(np.float32)
    return int(xs.max() - xs.min())


# ---------------------------------------------------------------------------
# TIBIA PCA — plateau angle
# ---------------------------------------------------------------------------
def plateau_slope_angle(poly):
    """
    Convex-hull plateau detection — robust to any image orientation.

    Method:
      1. Compute convex hull of the tibia mask contour.
      2. Find the LONGEST hull edge whose midpoint is in the upper 40%
         of the hull's y-range.  For an upright tibia this is the flat
         plateau top; for an oblique/tilted tibia it is the long diagonal
         cut surface — always the correct plateau edge regardless of rotation.
      3. Collect all hull points within 15px of that edge's line to get
         the full extent of the plateau (handles cases where the edge is
         split into several shorter collinear hull segments).
      4. Fit final line through those points for sub-pixel accuracy.

    Returns a Slope object:
      [0] = slope_deg (clamped ±30 to allow oblique views)
      [1] = (mask_cx, mask_cy) full-mask centroid
      .sl, .b      line coefficients  y = sl*x + b
      .band_xs     x-coords of plateau points (for width / cx)
      .band_cx     (x_min + x_max) / 2 of plateau points
    """
    pts  = poly.astype(np.float32)
    cnt  = pts.reshape(-1, 1, 2).astype(np.int32)
    hidx = cv2.convexHull(cnt, returnPoints=False).flatten()
    hull = pts[hidx]
    n    = len(hull)

    ys_h    = hull[:, 1]
    y_upper = float(ys_h.min()) + 0.40 * float(ys_h.max() - ys_h.min())
    cx_hull = float(hull[:, 0].mean())
    cy_hull = float(hull[:, 1].mean())

    # Find plateau edges via upward-normal + angle-consistency filter:
    # Step 1: collect hull edges in upper 40% whose outward normal points upward (ny<-0.50).
    # Step 2: compute length-weighted mean edge angle.
    # Step 3: discard edges >25° from that mean — removes anterior/posterior side
    #         faces that share an upward normal but belong to a different face.
    # Step 4: fit line through all surviving edge endpoints.
    candidates = []   # (length, edge_angle_deg, p1, p2)
    for i in range(n):
        p1, p2 = hull[i], hull[(i + 1) % n]
        if (p1[1] + p2[1]) / 2 > y_upper:
            continue
        l = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if l < 5:
            continue
        dx_e, dy_e = float(p2[0] - p1[0]), float(p2[1] - p1[1])
        nx, ny = dy_e, -dx_e
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        if nx * (cx_hull - mid_x) + ny * (cy_hull - mid_y) > 0:
            nx, ny = -nx, -ny
        nm = float(np.hypot(nx, ny))
        ny_norm = ny / nm if nm > 0 else 0.0
        if ny_norm < -0.50:
            edge_angle = float(np.degrees(np.arctan2(dy_e, dx_e)))
            candidates.append((l, edge_angle, p1, p2))

    if candidates:
        total_l  = sum(c[0] for c in candidates)
        w_angle  = sum(c[0] * c[1] for c in candidates) / total_l
        plat_pts = []
        for l, ea, p1, p2 in candidates:
            if abs(ea - w_angle) < 25.0:
                plat_pts.extend([p1, p2])
        if len(plat_pts) < 2:
            plat_pts = [p for c in candidates for p in (c[2], c[3])]
    else:
        plat_pts = []

    if len(plat_pts) < 2:
        top_idx  = int(np.argmin(hull[:, 1]))
        plat_pts = [hull[top_idx], hull[(top_idx + 1) % n]]

    pp = np.array(plat_pts)
    xb = pp[:, 0].astype(np.float64)
    yb = pp[:, 1].astype(np.float64)
    if xb.max() - xb.min() < 3.0:
        sl2, b2 = 0.0, float(yb.mean())
    else:
        sl2, b2 = np.polyfit(xb, yb, 1)

    angle  = float(np.clip(np.degrees(np.arctan(sl2)), -30.0, 30.0))
    mask_cx = float(pts[:, 0].mean())
    mask_cy = float(pts[:, 1].mean())

    class Slope:
        def __getitem__(self, i): return (angle, (mask_cx, mask_cy))[i]
        def __iter__(self):       return iter((angle, (mask_cx, mask_cy)))

    result          = Slope()
    result.sl       = sl2
    result.b        = b2
    result.band_xs  = xb
    result.band_cx  = float((xb.min() + xb.max()) / 2)
    return result


def poly_top_center(poly):
    """Top-center of tibia polygon (top 8% of points)."""
    ys = poly[:, 1].astype(float)
    min_y = ys.min()
    top = poly[ys <= min_y + 0.08 * (ys.max() - min_y)]
    return int((top[:, 0].min() + top[:, 0].max()) / 2), int(top[:, 1].mean())


# ---------------------------------------------------------------------------
# TIBIAL SHAFT AXIS
# ---------------------------------------------------------------------------
def tibial_shaft_axis(poly, H, W):
    """
    Find the tibial shaft axis using the CENTERLINE method:
    For each y-slice in the lower 50% of the tibia mask, compute the mean x
    of all contour points at that y level. Polyfit these (x_mean, y) pairs to
    get the true shaft lean direction — avoids PCA measuring the outline width.

    Returns:
      shaft_angle_deg : angle of shaft from horizontal (deg). ~90 = vertical shaft.
      line_pts        : ((x0,y0),(x1,y1)) endpoints extended to image bounds, or None.
    """
    pts = poly.astype(np.float32)
    ys  = pts[:, 1]
    y_min, y_max = float(ys.min()), float(ys.max())
    mask_height = y_max - y_min

    # Need the shaft to be visible: require mask covers >25% of image height.
    # Short masks (proximal tibia only) have no real shaft data — centerline
    # of a wide condyle blob gives a spurious near-horizontal "shaft" direction.
    if mask_height < H * 0.25:
        return None, None

    lower_thr = y_min + 0.50 * (y_max - y_min)

    # Build centerline: bin y into slices, take mean x per slice
    shaft_pts = pts[ys >= lower_thr]
    if shaft_pts.shape[0] < 8:
        return None, None

    n_bins = 20
    sy = shaft_pts[:, 1]
    bin_edges = np.linspace(sy.min(), sy.max(), n_bins + 1)
    cx_list, cy_list = [], []
    for i in range(n_bins):
        mask = (sy >= bin_edges[i]) & (sy < bin_edges[i + 1])
        if mask.sum() >= 2:
            cx_list.append(float(shaft_pts[:, 0][mask].mean()))
            cy_list.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))

    if len(cx_list) < 4:
        return None, None

    cx_arr = np.array(cx_list, dtype=np.float64)
    cy_arr = np.array(cy_list, dtype=np.float64)

    # Fit x = m*y + b  (y is the independent var since shaft is nearly vertical)
    m, b = np.polyfit(cy_arr, cx_arr, 1)   # x = m*y + b
    # Convert to angle from horizontal: dx/dy = m → angle = arctan2(1, m) from horizontal
    # shaft points downward: direction vector (dx, dy) = (m, 1) (normalised)
    shaft_angle_deg = float(np.degrees(np.arctan2(1.0, m)))  # ~90 for vertical shaft

    # Centroid of shaft region for line origin
    mean_cx = float(cx_arr.mean())
    mean_cy = float(cy_arr.mean())

    # Extend line to image bounds using parametric form
    # Point on line: (mean_cx, mean_cy), direction: (m, 1) normalised
    dx, dy = float(m), 1.0
    endpoints = []
    for t_val in [
        (0   - mean_cy) / dy if abs(dy) > 1e-6 else None,
        (H   - mean_cy) / dy if abs(dy) > 1e-6 else None,
        (0   - mean_cx) / dx if abs(dx) > 1e-6 else None,
        (W   - mean_cx) / dx if abs(dx) > 1e-6 else None,
    ]:
        if t_val is None:
            continue
        px = mean_cx + t_val * dx
        py = mean_cy + t_val * dy
        if -10 <= px <= W + 10 and -10 <= py <= H + 10:
            endpoints.append((int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))))

    if len(endpoints) < 2:
        return shaft_angle_deg, None

    best = max(
        [(endpoints[i], endpoints[j])
         for i in range(len(endpoints))
         for j in range(i + 1, len(endpoints))],
        key=lambda p: (p[0][0] - p[1][0])**2 + (p[0][1] - p[1][1])**2
    )
    return shaft_angle_deg, best


# ---------------------------------------------------------------------------
# PATELLA DETECTION → leg_mirror
# ---------------------------------------------------------------------------
def leg_mirror_from_patella_poly(p_poly, f_poly):
    """
    Use YOLO patella segmentation to determine which side anterior is on.
    Returns (leg_mirror: bool, pat_side: int, source: str)
      leg_mirror=False → standard (right knee, patella LEFT of femur)
      leg_mirror=True  → mirrored (left knee, patella RIGHT of femur)
      pat_side: +1 left, −1 right, 0 unknown
    """
    if p_poly is None or f_poly is None:
        return None, 0, "no_patella"
    pc = poly_centroid(p_poly)
    fc = poly_centroid(f_poly)
    if pc is None or fc is None:
        return None, 0, "centroid_fail"
    pat_side = +1 if pc[0] < fc[0] else -1
    # patella LEFT of femur (pat_side=+1) → standard right knee → leg_mirror=False
    return (pat_side == -1), pat_side, "patella_class"


def leg_mirror_heuristic_pixel(img_bgr, fem_bbox):
    """
    Fallback: look for bright bone blob anterior to femur.
    Returns (pat_side: int)  −1=right +1=left 0=uncertain
    """
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = fem_bbox
    cx = (x0 + x1) // 2
    pad_x = int((x1 - x0) * PATELLA_ROI_X_PAD_FRAC)
    roi_x0 = max(0, x0 - pad_x); roi_x1 = min(W - 1, x1 + pad_x)
    roi_y0 = max(0, int(y0 - (y1 - y0) * PATELLA_ROI_Y_FRAC_TOP))
    roi_y1 = min(H - 1, int(y0 + (y1 - y0) * PATELLA_ROI_Y_FRAC_BOT))
    if roi_x1 <= roi_x0 + 10 or roi_y1 <= roi_y0 + 10:
        return 0
    roi = img_bgr[roi_y0:roi_y1, roi_x0:roi_x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = float(np.quantile(g.astype(np.float32) / 255.0, PATELLA_THRESH_PCT) * 255.0)
    _, bw = cv2.threshold(g, int(thr), 255, cv2.THRESH_BINARY)
    bw = cv2.medianBlur(bw, 5)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return 0
    roi_area = float((roi_x1 - roi_x0) * (roi_y1 - roi_y0))
    best = max(cs, key=cv2.contourArea)
    if float(cv2.contourArea(best)) < PATELLA_MIN_AREA_FRAC * roi_area: return 0
    M = cv2.moments(best)
    if M["m00"] == 0: return 0
    bx = int(M["m10"] / M["m00"]) + roi_x0
    if abs(bx - cx) < int(0.05 * (x1 - x0)): return 0
    return -1 if bx < cx else +1


# ---------------------------------------------------------------------------
# GEOMETRY TRANSFORMS
# ---------------------------------------------------------------------------
def mirror_x(pts_n):
    out = pts_n.copy(); out[:, 0] = 1.0 - out[:, 0]; return out


def mirror_pt(pt_n):
    return np.array([1.0 - pt_n[0], pt_n[1]], dtype=np.float32)


def rotate_about(pts_n, anchor_n, deg):
    if abs(deg) < 1e-6: return pts_n
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    p = pts_n.astype(np.float32) - anchor_n.astype(np.float32)
    return (p @ R.T) + anchor_n.astype(np.float32)


def place_poly(pts_n, scale_px, anchor_n, anchor_px):
    pts = pts_n.astype(np.float32) * float(scale_px)
    off = np.array(anchor_px, dtype=np.float32) - (anchor_n.astype(np.float32) * float(scale_px))
    return (pts + off).astype(np.int32)


def draw_poly(img, pts_px, color, thick=3, closed=True):
    cv2.polylines(img, [pts_px], closed, color, thick, cv2.LINE_AA)


def draw_filled(img, pts_px, color, alpha=0.18):
    ov = img.copy(); cv2.fillPoly(ov, [pts_px], color)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def put_text(img, lines, x=20, y=36, dy=34):
    for i, t in enumerate(lines):
        yy = y + i * dy
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20,20,20),    1, cv2.LINE_AA)


def draw_size_box(img, fem_l, fem_mm, fem_diff, fem_ok,
                       tib_l, tib_mm, tib_diff, tib_ok,
                       mm_per_px, ps_source):
    H, W = img.shape[:2]
    fl = (f"FEMUR: Size {fem_l}   {fem_mm:.1f}mm (+/-{fem_diff:.1f}mm)" if fem_ok
          else f"FEMUR: OUT OF RANGE  {fem_mm:.1f}mm  (check calibration)")
    fc = (80,210,255) if fem_ok else (60,60,255)
    tl = (f"TIBIA: Size {tib_l}   {tib_mm:.1f}mm (+/-{tib_diff:.1f}mm)" if tib_ok
          else f"TIBIA: OUT OF RANGE  {tib_mm:.1f}mm  (check calibration)")
    tc = (80,230,130) if tib_ok else (60,60,255)
    px_per_mm = 1.0 / mm_per_px if mm_per_px > 1e-9 else 0.0
    sw = "" if ps_source == "DICOM" else f" [{ps_source}]"
    sl = f"mm/px: {mm_per_px:.4f}   px/mm: {px_per_mm:.3f}{sw}   NOT for clinical use"
    lines = [(fl, fc), (tl, tc), (sl, (200,200,200))]
    bh = len(lines) * 52 + 24; bw = min(900, W - 30); y0 = H - bh - 20
    ov = img.copy(); cv2.rectangle(ov, (15,y0), (15+bw, y0+bh), (0,0,0), -1)
    cv2.addWeighted(ov, 0.60, img, 0.40, 0, img)
    for i, (txt, col) in enumerate(lines):
        yp = y0 + 44 + i*52
        cv2.putText(img, txt, (25,yp), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(img, txt, (25,yp), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col,    2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def log_rec(out_dir, rec):
    os.makedirs(out_dir, exist_ok=True)
    cp = os.path.join(out_dir, "overlay_transforms.csv"); exists = os.path.exists(cp)
    with open(cp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if not exists: w.writeheader()
        w.writerow(rec)
    with open(os.path.join(out_dir, "overlay_transforms.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# GEOMETRY DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    fem_outer:         np.ndarray
    fem_anchor:        np.ndarray
    fem_anterior_hint: np.ndarray   # point on crescent/anterior edge in norm space

    tib_outer:  np.ndarray
    tib_insert: np.ndarray
    tib_keel_l: np.ndarray
    tib_keel_r: np.ndarray
    tib_anchor: np.ndarray          # original anchor (kept for compatibility)


# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------
def render_one(image_path, label_path, out_path, geom, default_mm_per_px,
               debug, enforce_femur_anterior):

    data = np.fromfile(image_path, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Cannot decode: {image_path}")
    H, W = img.shape[:2]

    polys   = read_polys(label_path, W, H)
    qc_err  = []; qc_warn = []

    rec = {
        "uid":              os.path.splitext(os.path.basename(image_path))[0],
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "mm_per_px":       "", "ps_source":        "",
        "fem_w_px":        "", "fem_AP_mm":        "", "fem_size":      "",
        "tib_LAP_mm":      "", "tib_size":         "",
        "fem_in_range":    "", "tib_in_range":     "",
        "plateau_slope_deg": "", "leg_mirror":     "",
        "patella_side":    "", "pat_source":       "",
        "fem_anterior_flip": "", "fem_rot_deg":    "",
        "tib_rot_deg":     "", "posterior_slope_deg": "", "qc_flags": "",
    }

    t_polys = [p for c, p in polys if c == CLASS_TIBIA]
    f_polys = [p for c, p in polys if c == CLASS_FEMUR]
    p_polys = [p for c, p in polys if c == CLASS_PATELLA]

    if not t_polys: qc_err.append("no_tibia")
    if not f_polys: qc_warn.append("no_femur")
    if not p_polys: qc_warn.append("no_patella")

    if debug:
        for c, p in polys:
            col = (0,255,0) if c==CLASS_FEMUR else (0,200,255) if c==CLASS_TIBIA else (0,200,255)
            cv2.polylines(img, [p], True, col, 2)
        # Draw patella in distinct color
        for c, p in polys:
            if c == CLASS_PATELLA:
                cv2.polylines(img, [p], True, (255, 180, 0), 2)  # orange-yellow

    def bail(msg):
        put_text(img, [msg])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img); rec["qc_flags"] = msg; return rec

    if not t_polys: return bail("QC:" + "|".join(qc_err))

    t_poly = biggest_poly(t_polys)

    # X-table detection is deferred until after shaft_angle is computed below.
    # (shaft_angle is a much more reliable indicator than bbox aspect ratio)
    f_poly = biggest_poly(f_polys) if f_polys else None
    p_poly = biggest_poly(p_polys) if p_polys else None

    if t_poly is None: return bail("QC:tib_none")

    # Auto-swap if tibia is above femur
    if f_poly is not None:
        tc = poly_centroid(t_poly); fc = poly_centroid(f_poly)
        if tc and fc and tc[1] < fc[1]:
            qc_warn.append("swapped"); t_poly, f_poly = f_poly, t_poly

    # Pixel spacing
    mm_per_px, ps_source = get_pixel_spacing(image_path, default_mm_per_px)
    rec["mm_per_px"] = round(mm_per_px, 4); rec["ps_source"] = ps_source

    # Scale + femur anchor
    if f_poly is not None:
        fx0, fy0, fx1, fy1 = poly_bbox(f_poly)
        fem_w_px    = femur_condyle_width(f_poly)
        fem_bbox_cx = (fx0 + fx1) // 2
        # Use 90th percentile y instead of bbox max — avoids noisy joint-line pixels
        # that extend the mask into the tibia region and pull the anchor too low.
        # 90th (vs 95th) keeps the femur slightly higher, reducing overlap with tray.
        fem_bot_y   = int(np.percentile(f_poly[:, 1].astype(np.float32), 90))
        fem_bbox    = (fx0, fy0, fx1, fy1)
        # Femur rotation: bottom-25% condyle edge slope (clamped ±12°)
        # 25% gives more stable polyfit than 15% on noisy YOLO contours.
        f_ys = f_poly[:, 1].astype(np.float32)
        bot_thr = float(f_ys.min()) + 0.75 * float(f_ys.max() - f_ys.min())
        bot_pts = f_poly[f_ys >= bot_thr].astype(np.float64)
        if bot_pts.shape[0] >= 6:
            b_sl, _ = np.polyfit(bot_pts[:, 0], bot_pts[:, 1], 1)
            fem_rot_from_mask = float(np.clip(np.degrees(np.arctan(b_sl)), -12.0, 12.0))
        else:
            fem_rot_from_mask = 0.0
    else:
        tx0, ty0, tx1, ty1 = poly_bbox(t_poly)
        fem_w_px    = int((tx1 - tx0) * (64.0 / 49.0))
        fem_bbox_cx = (tx0 + tx1) // 2
        fem_bot_y   = ty0
        fem_bbox    = (tx0, ty0, tx1, ty1)
        fem_rot_from_mask = 0.0
        qc_warn.append("fem_estimated")

    fem_scale  = float(fem_w_px) * SCALE_FRAC

    # Derive tib_scale directly from the YOLO tibia mask plateau width.
    # Use the top-15% of the mask (plateau level only — avoids the wider
    # fibula/soft-tissue region lower down the mask).
    _tp       = t_poly.astype(np.float32)
    _tp_ys    = _tp[:, 1]
    _tp_thr   = float(_tp_ys.min()) + 0.15 * float(_tp_ys.max() - _tp_ys.min())
    _plat_pts = _tp[_tp_ys <= _tp_thr]
    if _plat_pts.shape[0] < 2:
        _plat_pts = _tp                        # fallback: use full mask
    plateau_width_px  = float(_plat_pts[:, 0].max() - _plat_pts[:, 0].min())
    plateau_cx_mask   = float((_plat_pts[:, 0].max() + _plat_pts[:, 0].min()) / 2)

    # Normalised outer-shell x-span (computed from template after mirroring below,
    # but we need a preliminary value now — use the pre-mirrored span; mirroring
    # preserves the span so this is always correct).
    _tib_outer_tmp = geom.tib_outer.copy()
    _outer_x_span  = float(_tib_outer_tmp[:, 0].max() - _tib_outer_tmp[:, 0].min())
    if _outer_x_span < 0.01:
        _outer_x_span = 1.0                    # guard against degenerate template

    # Scale so that outer shell width == plateau_width_px
    tib_scale = plateau_width_px / _outer_x_span
    fem_AP_mm  = float(fem_w_px) * mm_per_px
    tib_LAP_mm = float(fem_w_px) * TIB_FROM_FEM_RATIO * mm_per_px

    fem_l, _, fem_diff, fem_ok = snap_femur(fem_AP_mm)
    tib_l, _, tib_diff, tib_ok = snap_tibia(tib_LAP_mm)
    if not fem_ok: qc_warn.append("fem_OOR")
    if not tib_ok: qc_warn.append("tib_OOR")

    rec.update({"fem_w_px": fem_w_px, "fem_AP_mm": round(fem_AP_mm, 1), "fem_size": fem_l,
                "tib_LAP_mm": round(tib_LAP_mm, 1), "tib_size": tib_l,
                "fem_in_range": fem_ok, "tib_in_range": tib_ok})

    # ---------------------------------------------------------------------------
    # [KEY FIX 1] leg_mirror from patella class
    # ---------------------------------------------------------------------------
    leg_mirror, pat_side, pat_source = leg_mirror_from_patella_poly(p_poly, f_poly)

    if leg_mirror is None:
        # Patella not detected — fall back to pixel heuristic
        pat_side_heuristic = leg_mirror_heuristic_pixel(img, fem_bbox)
        if pat_side_heuristic != 0:
            leg_mirror = (pat_side_heuristic == -1)
            pat_side   = pat_side_heuristic
            pat_source = "heuristic"
        else:
            # Last resort: femur position
            leg_mirror = fem_bbox_cx > (W / 2)
            pat_side   = 0
            pat_source = "position_fallback"
        qc_warn.append(f"pat_fallback:{pat_source}")

    rec["leg_mirror"]   = int(leg_mirror)
    rec["patella_side"] = pat_side
    rec["pat_source"]   = pat_source
    if leg_mirror: qc_warn.append("leg_mirrored")

    shared_cx = fem_bbox_cx

    # ---------------------------------------------------------------------------
    # [KEY FIX 2] Tibia PCA rotation
    # ---------------------------------------------------------------------------
    slope_result               = plateau_slope_angle(t_poly.astype(np.float32))
    slope_deg, tib_centroid_xy = slope_result[0], slope_result[1]
    rec["plateau_slope_deg"]   = round(slope_deg, 1)

    # Tibial shaft axis — centerline method, used for posterior slope + visual line
    shaft_angle_deg, shaft_line_pts = tibial_shaft_axis(t_poly, H, W)
    if shaft_angle_deg is not None:
        # shaft_angle_deg ≈ 90° for a vertical tibia.
        # X-table / rotated image: shaft deviates >40° from vertical (i.e. <50° or >130°)
        shaft_dev = abs(shaft_angle_deg - 90.0)
        if shaft_dev > 40.0:
            qc_warn.append("x_table_rotation")

        # Posterior slope = plateau angle relative to perpendicular-to-shaft
        # perp_to_shaft = shaft_angle_deg - 90  (in image horizontal terms)
        perp_to_shaft = shaft_angle_deg - 90.0
        posterior_slope_deg = float(np.clip(
            np.round(slope_deg - perp_to_shaft, 1), -15.0, 15.0))
    else:
        posterior_slope_deg = None
    rec["posterior_slope_deg"] = round(posterior_slope_deg, 1) if posterior_slope_deg is not None else ""

    # Rotation formula derived from template geometry (see module docstring):
    #   Non-mirror: tib_rot = slope_deg - TIB_TEMPLATE_ANGLE_DEG  = slope_deg + 18.23
    #   Mirror:     tib_rot = slope_deg + TIB_TEMPLATE_ANGLE_DEG  = slope_deg - 18.23
    if leg_mirror:
        tib_rot_deg = slope_deg + TIB_TEMPLATE_ANGLE_DEG
    else:
        tib_rot_deg = slope_deg - TIB_TEMPLATE_ANGLE_DEG

    tib_rot_deg = float(np.clip(tib_rot_deg, -30.0, 30.0))
    rec["tib_rot_deg"] = round(tib_rot_deg, 2)

    # ---------------------------------------------------------------------------
    # Anchors
    # ---------------------------------------------------------------------------
    fem_anchor_px = (shared_cx, fem_bot_y)

    # Reuse plateau line from slope_result (two-pass, rotation-invariant).
    _t_sl = slope_result.sl
    _t_b  = slope_result.b

    # Plateau width and centre-x from the band (rotation-invariant)
    _bxs = slope_result.band_xs
    if len(_bxs) >= 2:
        plateau_width_px = float(_bxs.max() - _bxs.min())
        plateau_cx_mask  = float(slope_result.band_cx)
    else:
        _tp = t_poly.astype(np.float32)
        plateau_width_px = float(_tp[:,0].max() - _tp[:,0].min())
        plateau_cx_mask  = float((_tp[:,0].max() + _tp[:,0].min()) / 2)

    # Y at the horizontal centre of the plateau → snap target
    plateau_cx_y = int(_t_sl * plateau_cx_mask + _t_b)

    # Debug line endpoints across the band width
    _t_x0 = int(_bxs.min()) if len(_bxs) >= 2 else 0
    _t_x1 = int(_bxs.max()) if len(_bxs) >= 2 else W
    plateau_line = (_t_x0, int(_t_sl*_t_x0+_t_b), _t_x1, int(_t_sl*_t_x1+_t_b))

    # tib_scale: match tray TOP EDGE LENGTH to plateau diagonal edge length.
    # plateau_width_px is the horizontal x-span of the band points.
    # Dividing by cos(slope) converts horizontal span → true diagonal edge length,
    # so the tray edge fills the plateau completely even at steep angles.
    _cos_slope = float(np.cos(np.radians(slope_deg)))
    if abs(_cos_slope) < 0.1:
        _cos_slope = 0.1   # guard: never divide by near-zero
    plateau_edge_len = plateau_width_px / _cos_slope
    tib_scale        = plateau_edge_len / _outer_x_span
    tib_anchor_px = (int(plateau_cx_mask), plateau_cx_y)

    # ---------------------------------------------------------------------------
    # Femur geometry — mirror then anterior-flip if needed
    # ---------------------------------------------------------------------------
    fem_outer_n  = geom.fem_outer.copy()
    fem_anchor_n = geom.fem_anchor.copy()

    if leg_mirror:
        fem_outer_n  = mirror_x(fem_outer_n)
        fem_anchor_n = mirror_pt(fem_anchor_n)

    # Anterior enforcement: make sure crescent faces patella side
    fem_anterior_flip = False
    if enforce_femur_anterior and pat_side != 0:
        hint_n = geom.fem_anterior_hint.copy()
        if leg_mirror:
            hint_n = mirror_pt(hint_n)
        # Place hint to see which side it lands on
        hint_px = place_poly(hint_n.reshape(1, 2), fem_scale, fem_anchor_n, fem_anchor_px)[0]
        hint_side = +1 if hint_px[0] < shared_cx else -1
        # pat_side: +1=patella LEFT, -1=patella RIGHT
        # We want anterior (hint) on same side as patella
        if hint_side != pat_side:
            fem_outer_n  = mirror_x(fem_outer_n)
            fem_anchor_n = mirror_pt(fem_anchor_n)
            fem_anterior_flip = True

    rec["fem_anterior_flip"] = int(fem_anterior_flip)

    # Use condyle bottom-edge slope as femur rotation (measured from YOLO mask)
    fem_rot_deg = fem_rot_from_mask
    rec["fem_rot_deg"] = round(fem_rot_deg, 2)

    fem_outer_n  = rotate_about(fem_outer_n, fem_anchor_n, fem_rot_deg)
    fem_outer_px = place_poly(fem_outer_n, fem_scale, fem_anchor_n, fem_anchor_px)

    # ---------------------------------------------------------------------------
    # Tibia geometry — mirror then PCA rotate around INSERT_ANCHOR_N
    # ---------------------------------------------------------------------------
    tib_outer_n  = geom.tib_outer.copy()
    tib_insert_n = geom.tib_insert.copy()
    tib_keel_l_n = geom.tib_keel_l.copy()
    tib_keel_r_n = geom.tib_keel_r.copy()

    if leg_mirror:
        tib_outer_n  = mirror_x(tib_outer_n)
        tib_insert_n = mirror_x(tib_insert_n)
        tib_keel_l_n = mirror_x(tib_keel_l_n)
        tib_keel_r_n = mirror_x(tib_keel_r_n)

    # Rotation anchor = insert midpoint (not tib_anchor_norm)
    if leg_mirror:
        rot_anchor_n = mirror_pt(INSERT_ANCHOR_N)
    else:
        rot_anchor_n = INSERT_ANCHOR_N.copy()

    tib_outer_n  = rotate_about(tib_outer_n,  rot_anchor_n, tib_rot_deg)
    tib_insert_n = rotate_about(tib_insert_n, rot_anchor_n, tib_rot_deg)
    tib_keel_l_n = rotate_about(tib_keel_l_n, rot_anchor_n, tib_rot_deg)
    tib_keel_r_n = rotate_about(tib_keel_r_n, rot_anchor_n, tib_rot_deg)

    # Place using insert anchor → tib_anchor_px (plateau centre point)
    tib_outer_px  = place_poly(tib_outer_n,  tib_scale, rot_anchor_n, tib_anchor_px)
    tib_insert_px = place_poly(tib_insert_n, tib_scale, rot_anchor_n, tib_anchor_px)
    tib_keel_l_px = place_poly(tib_keel_l_n, tib_scale, rot_anchor_n, tib_anchor_px)
    tib_keel_r_px = place_poly(tib_keel_r_n, tib_scale, rot_anchor_n, tib_anchor_px)

    # SNAP: shift every tib point DOWN so the tray TOP EDGE sits exactly on the plateau.
    # The insert midpoint (INSERT_ANCHOR_N[1] ≈ 0.274 into the template) was placed at
    # plateau_cx_y, which leaves the tray top floating ~0.274*scale px above the plateau.
    # We measure the actual top of the rotated outer shell and close that gap.
    # Snap: find the TOPMOST point of the outer shell within ±20% of tray width
    # of the plateau centre x. This is the top-edge point at the centre —
    # robust for any rotation angle without accidentally grabbing bottom points.
    _cx_target  = float(plateau_cx_mask)
    _tray_half  = plateau_width_px * 0.20          # ±20% of width = centre band
    _in_band    = np.abs(tib_outer_px[:, 0] - _cx_target) < _tray_half
    if _in_band.sum() < 1:                          # fallback: use all points
        _in_band = np.ones(len(tib_outer_px), dtype=bool)
    _tray_cx_y  = int(tib_outer_px[_in_band, 1].min())   # topmost = min y in band
    snap_shift  = plateau_cx_y - _tray_cx_y        # positive = shift down onto plateau
    if snap_shift != 0:
        tib_outer_px[:,  1] += snap_shift
        tib_insert_px[:, 1] += snap_shift
        tib_keel_l_px[:, 1] += snap_shift
        tib_keel_r_px[:, 1] += snap_shift

    # ---------------------------------------------------------------------------
    # Debug markers — hidden in clean mode
    # ---------------------------------------------------------------------------
    if debug:
        # Tibial shaft axis line
        if shaft_line_pts is not None:
            shaft_overlay = img.copy()
            cv2.line(shaft_overlay, shaft_line_pts[0], shaft_line_pts[1], (255, 255, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(shaft_overlay, 0.45, img, 0.55, 0, img)

        # Plateau detection line with tick marks at each end
        _pl = plateau_line
        cv2.line(img, (_pl[0], _pl[1]), (_pl[2], _pl[3]), (0, 0, 0),     4, cv2.LINE_AA)
        cv2.line(img, (_pl[0], _pl[1]), (_pl[2], _pl[3]), (0, 255, 255), 2, cv2.LINE_AA)
        for _px, _py in [(_pl[0], _pl[1]), (_pl[2], _pl[3])]:
            cv2.line(img, (_px, _py - 8), (_px, _py + 8), (0, 0, 0),     3, cv2.LINE_AA)
            cv2.line(img, (_px, _py - 8), (_px, _py + 8), (0, 255, 255), 2, cv2.LINE_AA)

        # Anchor dots
        cv2.circle(img, fem_anchor_px, 7, (0, 220, 0), -1)
        cv2.circle(img, tib_anchor_px, 7, (0, 0, 255), -1)

        # Femur YOLO bounding box
        if f_poly is not None:
            fx0, fy0, fx1, fy1 = poly_bbox(f_poly)
            cv2.rectangle(img, (fx0, fy0), (fx1, fy1), (0, 180, 60), 1)

        # Patella centroid dot
        if p_poly is not None:
            pc = poly_centroid(p_poly)
            if pc: cv2.circle(img, pc, 8, (0, 200, 255), -1)

    # ---------------------------------------------------------------------------
    # Draw overlays  (always shown — clean and debug)
    # ---------------------------------------------------------------------------
    if not debug:
        draw_filled(img, fem_outer_px, (220, 80, 200), alpha=0.22)
        draw_filled(img, tib_outer_px, (60, 100, 255), alpha=0.22)

    draw_poly(img, fem_outer_px,  (255, 80, 220), thick=4, closed=True)
    draw_poly(img, tib_outer_px,  (60, 200, 255), thick=3, closed=True)
    draw_poly(img, tib_insert_px, (60, 140, 255), thick=2, closed=True)
    draw_poly(img, tib_keel_l_px, (100, 160, 255), thick=2, closed=False)
    draw_poly(img, tib_keel_r_px, (100, 160, 255), thick=2, closed=False)

    # X-table overlay warning
    if debug and "x_table_rotation" in qc_warn:
        cx_img, cy_img = W // 2, H // 2
        cv2.putText(img, "X-TABLE / ROTATED IMAGE", (cx_img - 300, cy_img),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(img, "X-TABLE / ROTATED IMAGE", (cx_img - 300, cy_img),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 80, 255), 3, cv2.LINE_AA)

    # ---------------------------------------------------------------------------
    # HUD
    # ---------------------------------------------------------------------------
    qc_s = "|".join(qc_err) if qc_err else "OK"
    w_s  = (" W:" + ",".join(qc_warn)) if qc_warn else ""

    if debug:
        put_text(img, [
            f"fem: w={fem_w_px}px ap={fem_AP_mm:.1f}mm  scale={fem_scale:.0f}"
            f"  mir={int(leg_mirror)}  antFlip={int(fem_anterior_flip)}",
            f"tib: ap={tib_LAP_mm:.1f}mm  scale={tib_scale:.0f}  platW={plateau_width_px:.0f}px  top={plateau_cx_y}"
            f"  slope={slope_deg:.1f}deg  tibRot={tib_rot_deg:+.1f}  snap={snap_shift:+d}px"
            + (f"  postSlope={posterior_slope_deg:+.1f}deg" if posterior_slope_deg is not None else "  postSlope=N/A(short_mask)"),
            f"mm/px={mm_per_px:.4f}  px/mm={1/mm_per_px:.3f} [{ps_source}]"
            f"  patSide={pat_side}({pat_source})",
            "QC:" + qc_s + w_s,
        ])
        draw_size_box(img, fem_l, fem_AP_mm, fem_diff, fem_ok,
                           tib_l, tib_LAP_mm, tib_diff, tib_ok, mm_per_px, ps_source)
    else:
        draw_size_box(img, fem_l, fem_AP_mm, fem_diff, fem_ok,
                           tib_l, tib_LAP_mm, tib_diff, tib_ok, mm_per_px, ps_source)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    rec["qc_flags"] = "OK" if not qc_err else "|".join(qc_err)
    return rec


# ---------------------------------------------------------------------------
# CLI / BATCH RUNNER
# ---------------------------------------------------------------------------



def main():
    args = build_arg_parser().parse_args()
    debug_mode  = not _CLEAN_MODE
    enforce_ant = bool(args.enforce_femur_anterior) and not bool(args.no_enforce_femur_anterior)
    print(f"_CLEAN_MODE={_CLEAN_MODE}  debug_mode={debug_mode}")

    clean_dir = args.out_dir_clean if args.out_dir_clean else os.path.join(args.out_dir, "clean")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(clean_dir,    exist_ok=True)
    print(f"Debug  dir: {args.out_dir}")
    print(f"Clean  dir: {clean_dir}")
    print(f"PNG    dir: {args.png_dir}  (exists={os.path.isdir(args.png_dir)})")
    print(f"Label  dir: {args.label_dir}  (exists={os.path.isdir(args.label_dir)})")
    print(f"Mode: {'CLEAN' if not debug_mode else 'DEBUG'}")

    # PixelSpacing sidecars
    if args.extract_pixelspacing and os.path.isdir(args.png_dir):
        existing_json = [f for f in os.listdir(args.png_dir) if f.endswith(".json")]
        if not existing_json:
            print("Extracting PixelSpacing from DICOMs...")
            n = extract_pixel_spacing_batch(args.dicom_dir, args.png_dir)
            print(f"Done: {n} sidecars.\n")
        else:
            print(f"Found {len(existing_json)} PixelSpacing sidecars.\n")

    # Load geometry
    fem = load_svg_json(args.svg_dir, args.fem_json)
    tib = load_svg_json(args.svg_dir, args.tib_json)

    if "anterior_hint_norm" not in fem:
        print("WARNING: femur_lat.json missing 'anterior_hint_norm'.")
        print("         Add it manually: open femur_lat.json, add")
        print('         "anterior_hint_norm": [0.5, 0.1]')
        print("         (adjust x,y to the crescent/anterior edge in normalized space)")
        print("         Defaulting to [0.5, 0.1] for now.\n")
        fem["anterior_hint_norm"] = [0.5, 0.1]

    for k in ["outer_shell", "anchor_norm"]:
        if k not in fem: raise KeyError(f"femur json missing: {k}")
    for k in ["outer_shell", "insert", "keel_left", "keel_right", "anchor_norm"]:
        if k not in tib: raise KeyError(f"tibia json missing: {k}")

    geom = Geometry(
        fem_outer         = fem["outer_shell"],
        fem_anchor        = np.array(fem["anchor_norm"], dtype=np.float32),
        fem_anterior_hint = np.array(fem["anterior_hint_norm"], dtype=np.float32),
        tib_outer  = tib["outer_shell"],
        tib_insert = tib["insert"],
        tib_keel_l = tib["keel_left"],
        tib_keel_r = tib["keel_right"],
        tib_anchor = np.array(tib["anchor_norm"], dtype=np.float32),
    )

    # Load model (LateralV4.pt)
    print(f"Loading model: {args.model}")
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"Model classes: {model.names}")
        # Verify patella class index
        pat_idx = None
        for k, v in model.names.items():
            if "patella" in v.lower() or "pat" in v.lower():
                pat_idx = k
                break
        if pat_idx is not None and pat_idx != CLASS_PATELLA:
            print(f"WARNING: Patella is class {pat_idx} but CLASS_PATELLA={CLASS_PATELLA}.")
            print(f"         Edit CLASS_PATELLA at top of script to match.")
        use_yolo = True
    except Exception as e:
        print(f"WARNING: Could not load YOLO model ({e}). Using pre-existing label files.")
        use_yolo = False

    # Batch
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_files = [f for f in os.listdir(args.png_dir)
                 if os.path.isfile(os.path.join(args.png_dir, f))
                 and os.path.splitext(f)[1].lower() in img_exts]

    written = total = ok = no_label = fails = 0

    BLANK = {k: "" for k in [
        "uid","timestamp","mm_per_px","ps_source","fem_w_px","fem_AP_mm","fem_size",
        "tib_LAP_mm","tib_size","fem_in_range","tib_in_range","plateau_slope_deg",
        "leg_mirror","patella_side","pat_source","fem_anterior_flip","fem_rot_deg",
        "tib_rot_deg","qc_flags"
    ]}

    for fname in img_files:
        if written >= int(args.max_images): break
        uid        = os.path.splitext(fname)[0]
        fpath      = os.path.join(args.png_dir,  fname)
        label_path = os.path.join(args.label_dir, uid + ".txt")
        _active_out_dir = args.out_dir if debug_mode else clean_dir
        out_path   = os.path.join(_active_out_dir, uid + ("_debug.png" if debug_mode else "_clean.png"))

        total += 1
        print(f"[{total}] {fname}  label={'YES' if os.path.exists(label_path) else 'MISSING'}")

        # Run YOLO inference → write label txt
        if use_yolo:
            try:
                data_tmp = np.fromfile(fpath, dtype=np.uint8)
                img_tmp  = cv2.imdecode(data_tmp, cv2.IMREAD_COLOR)
                Ht, Wt   = img_tmp.shape[:2]
                results  = model.predict(source=fpath, conf=args.conf, iou=args.iou,
                                         save=False, verbose=False)
                os.makedirs(args.label_dir, exist_ok=True)
                with open(label_path, "w") as lf:
                    for r in results:
                        if r.masks is None: continue
                        for seg, cls_id in zip(r.masks.xy, r.boxes.cls.int().tolist()):
                            seg_n = seg.astype(np.float32)
                            seg_n[:, 0] /= Wt; seg_n[:, 1] /= Ht
                            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in seg_n)
                            lf.write(f"{cls_id} {coords}\n")
            except Exception as e:
                qc_str = f"yolo_fail:{str(e).replace(chr(10),' ')}"
                log_rec(args.out_dir, {**BLANK, "uid": uid,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "qc_flags": qc_str})
                fails += 1; continue

        if not os.path.exists(label_path):
            log_rec(args.out_dir, {**BLANK, "uid": uid,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "qc_flags": "no_label"})
            no_label += 1; continue

        try:
            r = render_one(fpath, label_path, out_path, geom, float(args.default_mm_per_px),
                           debug_mode, enforce_ant)
            log_rec(args.out_dir, r)
            if r.get("qc_flags") == "OK": ok += 1
            if args.save_only_qc_ok and r.get("qc_flags") != "OK":
                if os.path.exists(out_path): os.remove(out_path)
            else:
                written += 1
                print(f"  → saved: {out_path}")
        except Exception as e:
            import traceback
            print(f"  !! EXCEPTION: {e}")
            traceback.print_exc()
            log_rec(args.out_dir, {**BLANK, "uid": uid,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "qc_flags": "exc:" + str(e).replace("\n", " ")})
            fails += 1; written += 1

    print(f"\nDONE  total={total}  qc_ok={ok}  no_label={no_label}"
          f"  fail={fails}  saved={written}")
    print(f"Outputs: {args.out_dir}")
    print(f"Flags: debug={debug_mode}  enforce_ant={enforce_ant}")


def main_clean():
    """Run directly with: python lat_debug_render_v4_2.py clean
    Writes SVG-only overlays to <out_dir>\\clean\\  """
    import sys
    # Inject --clean into argv so argparse sees it
    if "--clean" not in sys.argv:
        sys.argv.append("--clean")
    main()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        sys.argv.pop(1)          # remove 'clean' positional
        sys.argv.append("--clean")  # replace with proper flag
    main()