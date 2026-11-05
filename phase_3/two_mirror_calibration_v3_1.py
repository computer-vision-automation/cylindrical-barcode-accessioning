"""
Two-mirror calibration - Phase 3.1  (prototype; per-photo calibration)

Changes from v3
  * EDGE mode (default): click 2 points on each of the 4 straight glass edges;
    corners are computed as line intersections. This works with ROUNDED glass
    corners and lets you use only the visible part of each edge.
  * Loupe window: an 8x magnifier follows the cursor so you can click with
    sub-screen-pixel accuracy on full-resolution photos.
  * Output goes to a sub-folder named after the photo (one calibration per photo,
    because a hand-held camera moves between shots).
  * Records the measured gap at top/bottom (11 / 8 mm), flask OD/ID, cap diameter.

What is implemented
  - Independent homography + rectification of each mirror to width*px_per_mm x height*px_per_mm
    (default 100 x 150 mm -> 1000 x 1500 px, i.e. 10 px/mm output resolution).
  - mm grid, reference/barcode points, dx/dy/distance in the mirror-plane coordinates.
  - JSON record with homographies, edge points, corners, parameters.

NOT implemented (Phase 4)
  - Reflection geometry. Angle, gaps, flask sizes are only STORED.
  - Displacements are APPARENT mirror-plane positions, not true distances to the flask.
  - Not validated against a physical ruler. 10 px/mm is output resolution, not accuracy.

IMPORTANT - use the VISIBLE GLASS size
  If the white frame covers part of the glass, rectify the visible area and pass its
  real size with --width-mm / --height-mm. Do not assume 100 x 150 if it is smaller.

Edge mode - for each mirror click, in this order (2 points each, far apart,
on the straight part of the edge, not near rounded corners):
    1) TOP edge   2) RIGHT edge   3) BOTTOM edge   4) LEFT edge
(top/right/bottom/left as seen in the PHOTO). Corner mode (--mode corner) instead
takes 4 clicks: TL, TR, BR, BL.

Run (PowerShell, from D:\\tech\\phase1):
  python two_mirror_calibration_v3_1.py samples/IMG_20261102_103828218.jpg

Controls
  Main window: 1 left mirror | 2 right mirror | click = add point | U undo | R reset
               Enter = rectify active mirror | S save | Q/Esc quit
  Mirror windows (hover first): M then click = reference | T then click = barcode
               C clear points | G grid on/off
"""

import argparse
import json
import math
import os
from datetime import datetime

import cv2
import numpy as np

MAIN_WINDOW = "Setup photo - click points"
LOUPE_WINDOW = "Loupe (8x)"
HEADER_H = 80
COLORS = {"left": (0, 220, 255), "right": (255, 160, 0)}  # BGR
SIDES = ("left", "right")
EDGE_NAMES = ("TOP", "RIGHT", "BOTTOM", "LEFT")


# ----------------------------------------------------------------------------
# Pure geometry helpers (no GUI)
# ----------------------------------------------------------------------------
def line_through(p, q):
    """Homogeneous line through two points."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def intersect(l1, l2):
    """Intersection of two homogeneous lines -> (x, y), or None if parallel."""
    v = np.cross(l1, l2)
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def corners_from_edges(edge_pts):
    """
    edge_pts: 8 points = [top1, top2, right1, right2, bottom1, bottom2, left1, left2].
    Returns corners TL, TR, BR, BL as intersections of neighbouring edge lines.
    """
    if len(edge_pts) != 8:
        return None
    top = line_through(edge_pts[0], edge_pts[1])
    right = line_through(edge_pts[2], edge_pts[3])
    bottom = line_through(edge_pts[4], edge_pts[5])
    left = line_through(edge_pts[6], edge_pts[7])
    pairs = ((top, left), (top, right), (bottom, right), (bottom, left))
    out = [intersect(a, b) for a, b in pairs]
    return None if any(c is None for c in out) else out


def is_valid_quad(pts):
    """True if 4 points form a convex quadrilateral."""
    p = np.asarray(pts, dtype=float)
    if p.shape != (4, 2):
        return False
    s = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        s.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(x > 0 for x in s) or all(x < 0 for x in s)


def compute_homography(corners_px, out_w, out_h):
    """H maps photo px -> rectified px:  p_rect ~ H @ p_photo."""
    src = np.asarray(corners_px, dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def apply_h(H, pt):
    v = H @ np.array([pt[0], pt[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def displacement_mm(ref, tgt, px_per_mm):
    dx = (tgt[0] - ref[0]) / px_per_mm
    dy = (tgt[1] - ref[1]) / px_per_mm
    return dx, dy, float(np.hypot(dx, dy))


def gap_wedge_deg(gap_top_mm, gap_bottom_mm, height_mm):
    """Tilt implied by a gap that differs between top and bottom (small-angle geometry)."""
    return math.degrees(math.atan2(gap_top_mm - gap_bottom_mm, height_mm))


def make_loupe(image, center, points, half=24, zoom=8):
    """Magnified crop around `center` (image coords) with a crosshair and nearby points."""
    h, w = image.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    pad = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    crop = pad[cy:cy + 2 * half, cx:cx + 2 * half]
    if crop.shape[0] != 2 * half or crop.shape[1] != 2 * half:
        crop = np.full((2 * half, 2 * half, 3), 40, np.uint8)
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    for (px, py) in points:
        bx, by = (px - (cx - half) + 0.5) * zoom, (py - (cy - half) + 0.5) * zoom
        if 0 <= bx < big.shape[1] and 0 <= by < big.shape[0]:
            cv2.circle(big, (int(bx), int(by)), 6, (0, 0, 255), 2)
    c = half * zoom
    cv2.line(big, (c, 0), (c, 2 * half * zoom), (0, 255, 0), 1)
    cv2.line(big, (0, c), (2 * half * zoom, c), (0, 255, 0), 1)
    return big


def draw_mm_grid(img, width_mm, height_mm, px_per_mm):
    out = img.copy()
    overlay = img.copy()
    h, w = img.shape[:2]
    for mm in range(0, int(width_mm) + 1, 10):
        x = min(int(round(mm * px_per_mm)), w - 1)
        cv2.line(overlay, (x, 0), (x, h - 1), (255, 180, 0), 2 if mm % 50 == 0 else 1)
    for mm in range(0, int(height_mm) + 1, 10):
        y = min(int(round(mm * px_per_mm)), h - 1)
        cv2.line(overlay, (0, y), (w - 1, y), (255, 180, 0), 2 if mm % 50 == 0 else 1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    for mm in range(0, int(width_mm) + 1, 20):
        x = int(round(mm * px_per_mm))
        cv2.putText(out, f"{mm}", (min(x + 3, w - 40), 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
    for mm in range(20, int(height_mm) + 1, 20):
        y = int(round(mm * px_per_mm))
        cv2.putText(out, f"{mm}", (4, min(y - 4, h - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "mm", (w - 45, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------
class TwoMirrorCalibrator:
    def __init__(self, image, image_path, args):
        self.image = image
        self.image_path = image_path
        self.h, self.w = image.shape[:2]
        self.mode = args.mode
        self.n_clicks = 8 if self.mode == "edge" else 4

        self.width_mm, self.height_mm = args.width_mm, args.height_mm
        self.px_per_mm = args.px_per_mm
        self.out_w = int(round(self.width_mm * self.px_per_mm))
        self.out_h = int(round(self.height_mm * self.px_per_mm))
        self.params = {
            "included_angle_deg_nominal": args.angle_deg,
            "gap_top_mm_measured": args.gap_top_mm,
            "gap_bottom_mm_measured": args.gap_bottom_mm,
            "flask_outer_diameter_mm": args.flask_od_mm,
            "flask_inner_diameter_mm": args.flask_id_mm,
            "cap_diameter_mm": args.cap_diameter_mm,
            "flask_position_cm_label": args.flask_position_cm,
        }
        stem = os.path.splitext(os.path.basename(image_path))[0]
        self.output_dir = os.path.join(args.out_dir, stem)

        self.active = "left"
        self.points = {s: [] for s in SIDES}
        self.corners = {s: None for s in SIDES}
        self.H = {s: None for s in SIDES}
        self.rectified = {s: None for s in SIDES}
        self.meas = {s: {"reference": None, "target": None, "mode": None} for s in SIDES}
        self.windows = set()
        self.hover = None
        self.show_grid = True
        self.dirty = {s: True for s in SIDES}
        self.cursor = None
        self.cursor_dirty = False
        self.status = "Press 1 (left) or 2 (right) and start clicking."
        self.main_fit = min(1.0, 1300.0 / self.w, 850.0 / (self.h + HEADER_H))

    # ---------- prompts ----------
    def prompt(self):
        n = len(self.points[self.active])
        if n >= self.n_clicks:
            return "All points set - press Enter to rectify."
        if self.mode == "edge":
            return f"{self.active.upper()}: click 2 points on the {EDGE_NAMES[n // 2]} edge (point {n % 2 + 1}/2)."
        return f"{self.active.upper()}: click corner {('TL', 'TR', 'BR', 'BL')[n]} ({n + 1}/4)."

    # ---------- main window ----------
    def render_main(self):
        canvas = cv2.copyMakeBorder(self.image, HEADER_H, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        r = max(4, int(round(min(self.w, self.h) / 300)))
        for side in SIDES:
            pts, color = self.points[side], COLORS[side]
            shifted = [(x, y + HEADER_H) for x, y in pts]
            if self.mode == "edge":
                for e in range(len(shifted) // 2):
                    p, q = np.array(shifted[2 * e], float), np.array(shifted[2 * e + 1], float)
                    d = q - p
                    a, b = p - 30 * d, p + 30 * d
                    ok, a2, b2 = cv2.clipLine((0, 0, self.w, self.h + HEADER_H),
                                              (int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
                    if ok:
                        cv2.line(canvas, a2, b2, color, 1, cv2.LINE_AA)
            elif len(shifted) >= 2:
                cv2.polylines(canvas, [np.array(shifted, np.int32)], len(shifted) == 4,
                              color, max(1, r // 2), cv2.LINE_AA)
            for i, (x, y) in enumerate(shifted):
                cv2.circle(canvas, (x, y), r, color, -1, cv2.LINE_AA)
            c = self.corners[side]
            if c:
                for (x, y) in c:
                    cv2.drawMarker(canvas, (int(x), int(y) + HEADER_H), (0, 255, 0),
                                   cv2.MARKER_CROSS, 4 * r, 2)
        fs = max(0.6, self.w / 2200)
        cv2.putText(canvas, f"{self.prompt()}  {self.status if self.status != self.prompt() else ''}",
                    (10, int(HEADER_H * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "1 Left | 2 Right | U undo | R reset | Enter rectify | S save | Q quit   [mode: "
                    + self.mode + "]", (10, int(HEADER_H * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.75,
                    (210, 210, 210), 1, cv2.LINE_AA)
        return canvas

    def on_main_mouse(self, event, x, y, flags, param):
        y -= HEADER_H
        if event == cv2.EVENT_MOUSEMOVE:
            if y >= 0:
                self.cursor = (x, y)
                self.cursor_dirty = True
            return
        if event != cv2.EVENT_LBUTTONDOWN or y < 0:
            return
        pts = self.points[self.active]
        if len(pts) >= self.n_clicks:
            self.status = "Already complete: Enter to rectify, or U to undo."
            return
        pts.append((int(x), int(y)))
        self.status = ""

    # ---------- rectification ----------
    def rectify(self, side):
        pts = self.points[side]
        if len(pts) != self.n_clicks:
            self.status = f"{side.upper()}: need {self.n_clicks} points (have {len(pts)})."
            return False
        if self.mode == "edge":
            min_sep = 0.06 * max(self.w, self.h)
            for e in range(4):
                a, b = pts[2 * e], pts[2 * e + 1]
                d = math.hypot(a[0] - b[0], a[1] - b[1])
                if d < min_sep:
                    self.status = (f"{side.upper()}: {EDGE_NAMES[e]} edge points only {d:.0f}px apart "
                                   f"(need >= {min_sep:.0f}px). Spread them along the edge. Press R and redo.")
                    return False
        corners = corners_from_edges(pts) if self.mode == "edge" else [tuple(map(float, p)) for p in pts]
        if corners is None:
            self.status = f"{side.upper()}: two edges are parallel - redo the clicks."
            return False
        if not is_valid_quad(corners):
            self.status = f"{side.upper()}: corners not a convex quad - check edge order (TOP,RIGHT,BOTTOM,LEFT). Press R."
            return False
        self.corners[side] = corners
        self.H[side] = compute_homography(corners, self.out_w, self.out_h)
        self.rectified[side] = cv2.warpPerspective(self.image, self.H[side], (self.out_w, self.out_h))
        self.dirty[side] = True
        self.status = f"{side.upper()} rectified to {self.out_w}x{self.out_h}px ({self.px_per_mm:g} px/mm)."
        return True

    def invalidate(self, side):
        self.H[side] = self.rectified[side] = self.corners[side] = None
        if side in self.windows:
            try:
                cv2.destroyWindow(self.win_name(side))
            except cv2.error:
                pass
            self.windows.discard(side)

    def undo(self):
        if self.points[self.active]:
            self.points[self.active].pop()
            self.invalidate(self.active)
            self.status = ""

    def reset(self):
        self.points[self.active] = []
        self.invalidate(self.active)
        self.meas[self.active] = {"reference": None, "target": None, "mode": None}
        self.status = f"{self.active.upper()} reset."

    # ---------- rectified windows ----------
    @staticmethod
    def win_name(side):
        return f"{side.upper()} mirror - rectified (mm grid)"

    def ensure_window(self, side):
        if side in self.windows or self.rectified[side] is None:
            return
        name = self.win_name(side)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 400, int(400 * self.out_h / self.out_w))
        cv2.setMouseCallback(name, self.on_rect_mouse, side)
        self.windows.add(side)
        self.dirty[side] = True

    def on_rect_mouse(self, event, x, y, flags, side):
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = side
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.hover = side
        m = self.meas[side]
        if m["mode"] in ("reference", "target"):
            m[m["mode"]] = (int(x), int(y))
            print(f"[{side.upper()}] {m['mode']} = ({x / self.px_per_mm:.2f}, {y / self.px_per_mm:.2f}) mm from top-left")
            m["mode"] = None
            self.dirty[side] = True

    def render_rect(self, side):
        img = self.rectified[side]
        canvas = draw_mm_grid(img, self.width_mm, self.height_mm, self.px_per_mm) if self.show_grid else img.copy()
        m = self.meas[side]
        ref, tgt = m["reference"], m["target"]
        if ref:
            cv2.drawMarker(canvas, ref, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(canvas, "REF", (ref[0] + 12, ref[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if tgt:
            cv2.drawMarker(canvas, tgt, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(canvas, "BARCODE", (tgt[0] + 12, tgt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if ref and tgt:
            cv2.line(canvas, ref, tgt, (0, 255, 255), 2, cv2.LINE_AA)
            dx, dy, d = displacement_mm(ref, tgt, self.px_per_mm)
            cv2.rectangle(canvas, (0, self.out_h - 50), (self.out_w, self.out_h), (20, 20, 20), -1)
            cv2.putText(canvas, f"dx={dx:+.2f}mm dy={dy:+.2f}mm dist={d:.2f}mm (apparent)", (10, self.out_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if m["mode"]:
            cv2.putText(canvas, f"click to set {m['mode']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return canvas

    def measure_side(self):
        if self.hover in self.windows:
            return self.hover
        return self.active if self.rectified[self.active] is not None else None

    # ---------- saving ----------
    def build_record(self):
        rec = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "source_image": os.path.abspath(self.image_path),
            "image_size_px": {"width": self.w, "height": self.h},
            "corner_mode": self.mode,
            "mirror_visible_glass_mm": {"width": self.width_mm, "height": self.height_mm},
            "rectified_size_px": {"width": self.out_w, "height": self.out_h},
            "px_per_mm": self.px_per_mm,
            "geometry_stored_for_phase4_not_yet_used": dict(self.params),
            "derived": {
                "gap_wedge_tilt_deg": gap_wedge_deg(self.params["gap_top_mm_measured"],
                                                    self.params["gap_bottom_mm_measured"], self.height_mm),
                "note": "Gap differs top vs bottom: mirror planes are not exactly vertical/parallel to the hinge.",
            },
            "validated_against_physical_ruler": False,
            "notes": ["px_per_mm is output resolution, not accuracy.",
                      "Displacements are apparent mirror-plane positions, not true 3D distances."],
            "mirrors": {},
        }
        for side in SIDES:
            e = {"clicked_points_photo_px": [list(p) for p in self.points[side]],
                 "corners_photo_px_TL_TR_BR_BL": [list(c) for c in self.corners[side]] if self.corners[side] else None,
                 "homography_photo_to_rectified": None, "homography_rectified_to_photo": None,
                 "reference_point": None, "barcode_point": None, "displacement": None}
            if self.H[side] is not None:
                Hm = self.H[side]
                Hi = np.linalg.inv(Hm)
                e["homography_photo_to_rectified"] = Hm.tolist()
                e["homography_rectified_to_photo"] = Hi.tolist()
                ref, tgt = self.meas[side]["reference"], self.meas[side]["target"]
                for key, pt in (("reference_point", ref), ("barcode_point", tgt)):
                    if pt:
                        e[key] = {"rectified_px": list(pt),
                                  "mm_from_top_left": [pt[0] / self.px_per_mm, pt[1] / self.px_per_mm],
                                  "photo_px": list(apply_h(Hi, pt))}
                if ref and tgt:
                    dx, dy, d = displacement_mm(ref, tgt, self.px_per_mm)
                    e["displacement"] = {"dx_mm": dx, "dy_mm": dy, "distance_mm": d,
                                         "meaning": "apparent, mirror-plane coordinates"}
            rec["mirrors"][side] = e
        return rec

    def save(self):
        out = self.output_dir
        os.makedirs(out, exist_ok=True)
        for side in SIDES:
            if self.rectified[side] is not None:
                cv2.imwrite(os.path.join(out, f"{side}_mirror_calibrated.png"), self.render_rect(side))
                cv2.imwrite(os.path.join(out, f"{side}_mirror_rectified_raw.png"), self.rectified[side])
        summary = self.image.copy()
        r = max(4, int(round(min(self.w, self.h) / 300)))
        for side in SIDES:
            for p in self.points[side]:
                cv2.circle(summary, p, r, COLORS[side], -1, cv2.LINE_AA)
            c = self.corners[side]
            if c:
                cv2.polylines(summary, [np.array(c, np.int32)], True, COLORS[side], max(2, r // 2), cv2.LINE_AA)
        cv2.imwrite(os.path.join(out, "selected_mirror_corners.png"), summary)
        with open(os.path.join(out, "calibration_record.json"), "w", encoding="utf-8") as f:
            json.dump(self.build_record(), f, indent=2)
        self.status = f"Saved to {os.path.abspath(out)}"
        print(f"[SAVED] {os.path.abspath(out)}")
        for name in sorted(os.listdir(out)):
            print("   ", name)

    # ---------- loop ----------
    def run(self):
        cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(MAIN_WINDOW, int(self.w * self.main_fit), int((self.h + HEADER_H) * self.main_fit))
        cv2.setMouseCallback(MAIN_WINDOW, self.on_main_mouse)
        cv2.namedWindow(LOUPE_WINDOW, cv2.WINDOW_AUTOSIZE)
        print("\n--- Two Mirror Calibration v3.1 ---")
        print("1/2 pick mirror. Edge mode: 2 points on TOP, RIGHT, BOTTOM, LEFT edges. Enter = rectify. S = save.\n")
        while True:
            cv2.imshow(MAIN_WINDOW, self.render_main())
            if self.cursor and self.cursor_dirty:
                cv2.imshow(LOUPE_WINDOW, make_loupe(self.image, self.cursor, self.points[self.active]))
                self.cursor_dirty = False
            for side in SIDES:
                self.ensure_window(side)
                if side in self.windows and self.dirty[side]:
                    cv2.imshow(self.win_name(side), self.render_rect(side))
                    self.dirty[side] = False
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            k = chr(key).lower() if 32 <= key < 127 else key
            if key == 27 or k == "q":
                break
            elif k == "1":
                self.active, self.status = "left", ""
            elif k == "2":
                self.active, self.status = "right", ""
            elif k == "u":
                self.undo()
            elif k == "r":
                self.reset()
            elif key in (10, 13):
                self.rectify(self.active)
            elif k == "s":
                self.save()
            elif k == "g":
                self.show_grid = not self.show_grid
                self.dirty = {s: True for s in SIDES}
            elif k in ("m", "t", "c"):
                side = self.measure_side()
                if side is None:
                    self.status = "Rectify a mirror first (Enter)."
                    continue
                if k == "c":
                    self.meas[side] = {"reference": None, "target": None, "mode": None}
                else:
                    self.meas[side]["mode"] = "reference" if k == "m" else "target"
                    print(f"[{side.upper()}] click the {self.meas[side]['mode']} point in its window")
                self.dirty[side] = True
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Two-mirror perspective calibration, per photo (Phase 3.1).")
    p.add_argument("image", help="Path to a setup photo (full resolution)")
    p.add_argument("--mode", choices=("edge", "corner"), default="edge", help="edge = 8 clicks (default), corner = 4")
    p.add_argument("--width-mm", type=float, default=100.0, help="VISIBLE glass width (default 100)")
    p.add_argument("--height-mm", type=float, default=150.0, help="VISIBLE glass height (default 150)")
    p.add_argument("--px-per-mm", type=float, default=10.0, help="Rectified output resolution")
    p.add_argument("--angle-deg", type=float, default=99.0, help="Angle between mirrors (stored only)")
    p.add_argument("--gap-top-mm", type=float, default=11.0, help="Measured gap at top (stored)")
    p.add_argument("--gap-bottom-mm", type=float, default=8.0, help="Measured gap at bottom (stored)")
    p.add_argument("--flask-od-mm", type=float, default=13.0, help="Flask outer diameter (stored)")
    p.add_argument("--flask-id-mm", type=float, default=11.0, help="Flask inner diameter (stored)")
    p.add_argument("--cap-diameter-mm", type=float, default=15.0, help="Cap diameter (stored)")
    p.add_argument("--flask-position-cm", type=float, default=None, help="Flask distance on centre line, if known (label only)")
    p.add_argument("--out-dir", default="calibration_output_v3_1", help="Base output folder")
    args = p.parse_args()
    for n in ("width_mm", "height_mm", "px_per_mm"):
        if getattr(args, n) <= 0:
            p.error(f"--{n.replace('_', '-')} must be positive")
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}\nCheck path and extension.")
    TwoMirrorCalibrator(image, args.image, args).run()


if __name__ == "__main__":
    main()