"""
Batch two-mirror image measurement for the Phase 4 reflection model.

Calibrate mirror planes ONCE on the first photograph, then reuse the same
homographies for the four fixed-camera photographs. For each image, click
the centreline of the reflected tube in the LEFT and RIGHT rectified mirror.

Run from the folder containing this script, the four JPGs, and
reflection_model_2d_hinge_reference.py:

python phase4_batch_measure.py IMG_20261105_220736475.jpg IMG_20261105_220757834.jpg IMG_20261105_220812741.jpg IMG_20261105_220823603.jpg --positions-mm 30 50 70 100 --camera-y-mm 240

Mirror calibration clicks on FIRST photo:
For LEFT mirror, click 2 points on each edge in order:
TOP, RIGHT (hinge-side), BOTTOM, LEFT.
For RIGHT mirror:
TOP, RIGHT, BOTTOM, LEFT.
Choose points far apart along each straight edge, avoiding rounded corners.

Then each rectified image pair is shown side-by-side. Click the reflected
tube's vertical centreline at approximately mid-height in LEFT, then RIGHT.
Press U to undo last target click; Q/Esc aborts.

Important:
- The four photos must have the same camera position, orientation, zoom and
  mirror placement for reusing the first photo's homographies.
- The outputs are apparent mirror-plane positions, comparable to the model's
  predicted reflection-point distances from each hinge-side glass edge.
- This is not a 3D reconstruction or barcode decoding step.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

try:
    from reflection_model_2d_hinge_reference import axis_prediction
except Exception:
    axis_prediction = None

EDGE_NAMES = ("TOP", "RIGHT", "BOTTOM", "LEFT")
SIDES = ("left", "right")


def line_through(p, q):
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def intersect(l1, l2):
    v = np.cross(l1, l2)
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def corners_from_edges(points):
    top = line_through(points[0], points[1])
    right = line_through(points[2], points[3])
    bottom = line_through(points[4], points[5])
    left = line_through(points[6], points[7])
    out = [intersect(top, left), intersect(top, right),
           intersect(bottom, right), intersect(bottom, left)]
    if any(p is None for p in out):
        raise ValueError("Could not intersect mirror edges. Re-select points.")
    return np.asarray(out, np.float32)


class Clicker:
    def __init__(self, image, title, expected, scale=1.0):
        self.image = image
        self.title = title
        self.expected = expected
        self.scale = scale
        self.points = []
        self.done = False
        self.window = title
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.expected:
            self.points.append((x / self.scale, y / self.scale))

    def run(self, instructions):
        while True:
            view = cv2.resize(self.image, None, fx=self.scale, fy=self.scale)
            cv2.putText(view, instructions, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(self.points):
                cv2.circle(view, (int(x*self.scale), int(y*self.scale)), 6,
                           (0, 0, 255), -1)
                cv2.putText(view, str(i+1), (int(x*self.scale)+8, int(y*self.scale)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(self.window, view)
            key = cv2.waitKey(20) & 0xFF
            if len(self.points) == self.expected:
                break
            if key in (27, ord('q')):
                raise KeyboardInterrupt
            if key == ord('u') and self.points:
                self.points.pop()
        cv2.destroyWindow(self.window)
        return np.asarray(self.points, np.float32)


def choose_mirror_edges(image, side, scale):
    print(f"\nCalibrating {side.upper()} mirror: click two points on each edge.")
    c = Clicker(image, f"Select {side.upper()} mirror edges", 8, scale)
    pts = c.run(f"{side.upper()}: 2 points each - TOP, RIGHT, BOTTOM, LEFT")
    corners = corners_from_edges(pts)
    return corners


def make_rectifier(corners, width_mm, height_mm, px_per_mm):
    out_w = int(round(width_mm * px_per_mm))
    out_h = int(round(height_mm * px_per_mm))
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return H, (out_w, out_h)


def click_targets(pair, scale):
    """pair image: left rectified mirror on left, right rectified on right."""
    h, w = pair.shape[:2]
    c = Clicker(pair, "Click reflected tube centreline: LEFT then RIGHT", 2, scale)
    pts = c.run("Click LEFT reflected tube centreline, then RIGHT (mid-height)")
    # split at original rectified mirror width
    half = w // 2
    return (float(pts[0][0]), float(pts[1][0] - half))


def main():
    ap = argparse.ArgumentParser(description="Batch measure reflected tube axes in four fixed-camera images.")
    ap.add_argument("images", nargs="+", help="Images ordered to match positions, e.g. 3,5,7,10 cm")
    ap.add_argument("--positions-mm", nargs="+", type=float, required=True)
    ap.add_argument("--camera-y-mm", type=float, default=240.0)
    ap.add_argument("--camera-x-mm", type=float, default=0.0)
    ap.add_argument("--angle-deg", type=float, default=99.0)
    ap.add_argument("--gap-mm", type=float, default=9.5)
    ap.add_argument("--width-mm", type=float, default=100.0)
    ap.add_argument("--height-mm", type=float, default=150.0)
    ap.add_argument("--px-per-mm", type=float, default=10.0)
    ap.add_argument("--out-dir", default="phase4_measurements")
    args = ap.parse_args()

    if len(args.images) != len(args.positions_mm):
        ap.error("Number of images must equal number of --positions-mm values.")

    images = []
    for p in args.images:
        im = cv2.imread(p)
        if im is None:
            ap.error(f"Cannot open image: {p}")
        images.append(im)

    shape = images[0].shape
    if any(im.shape != shape for im in images):
        ap.error("All images must have identical dimensions for shared homographies.")

    os.makedirs(args.out_dir, exist_ok=True)
    scale = min(1.0, 1250.0 / shape[1], 850.0 / shape[0])

    print("\nUse the first image to calibrate both visible mirror glass planes.")
    print("Corner order is computed from edge lines; avoid rounded corners.")
    left_corners = choose_mirror_edges(images[0], "left", scale)
    right_corners = choose_mirror_edges(images[0], "right", scale)

    H_left, size = make_rectifier(left_corners, args.width_mm, args.height_mm, args.px_per_mm)
    H_right, _ = make_rectifier(right_corners, args.width_mm, args.height_mm, args.px_per_mm)
    out_w, out_h = size

    # Save the reusable geometry so it can be audited/reused.
    with open(os.path.join(args.out_dir, "mirror_homographies.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source_image": args.images[0],
            "left_corners_photo_px": left_corners.tolist(),
            "right_corners_photo_px": right_corners.tolist(),
            "left_photo_to_rectified": H_left.tolist(),
            "right_photo_to_rectified": H_right.tolist(),
            "mirror_width_mm": args.width_mm,
            "mirror_height_mm": args.height_mm,
            "px_per_mm": args.px_per_mm,
            "note": "Shared homographies assume fixed camera and unchanged mirror rig across all images."
        }, f, indent=2)

    rows = []
    for idx, (path, im, position) in enumerate(zip(args.images, images, args.positions_mm), 1):
        left = cv2.warpPerspective(im, H_left, (out_w, out_h))
        right = cv2.warpPerspective(im, H_right, (out_w, out_h))
        pair = np.hstack([left, right])
        pair_path = os.path.join(args.out_dir, f"{idx:02d}_{position:g}mm_rectified_pair.jpg")
        cv2.imwrite(pair_path, pair)

        # Downscale display while retaining full-res click coordinates.
        display_scale = min(1.0, 1500.0 / pair.shape[1], 850.0 / pair.shape[0])
        print(f"\nImage {idx}/{len(images)}: {path}  |  flask position {position:g} mm")
        x_left_px, x_right_px = click_targets(pair, display_scale)

        # Rectified x=0 is photo-left. Left mirror hinge edge is its RIGHT edge;
        # right mirror hinge edge is its LEFT edge.
        left_mm = (args.width_mm * args.px_per_mm - x_left_px) / args.px_per_mm
        right_mm = x_right_px / args.px_per_mm

        predL = predR = float("nan")
        if axis_prediction is not None:
            pred = axis_prediction(position, np.array([args.camera_x_mm, args.camera_y_mm]),
                                   args.angle_deg, args.gap_mm, args.width_mm)
            if pred["left"] is not None:
                predL = float(pred["left"]["d_edge"])
            if pred["right"] is not None:
                predR = float(pred["right"]["d_edge"])

        rows.append({
            "position_mm": position,
            "image": os.path.basename(path),
            "measured_left_mm": left_mm,
            "predicted_left_mm": predL,
            "left_error_mm": left_mm - predL,
            "measured_right_mm": right_mm,
            "predicted_right_mm": predR,
            "right_error_mm": right_mm - predR,
            "measured_asymmetry_L_minus_R_mm": left_mm - right_mm,
        })

    csv_path = os.path.join(args.out_dir, "phase4_measured_vs_predicted.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.out_dir, "phase4_measured_vs_predicted.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "camera_y_mm": args.camera_y_mm,
            "camera_x_mm": args.camera_x_mm,
            "angle_deg": args.angle_deg,
            "gap_mm": args.gap_mm,
            "positions_mm": args.positions_mm,
            "measurements": rows,
            "warning": "Manual centreline clicks and shared homography; inspect rectified pairs. These are apparent mirror-plane positions, not direct 3D flask coordinates."
        }, f, indent=2)

    print("\nRESULTS (apparent mirror-plane positions, measured from hinge-side glass edge)")
    print("position | measured L | predicted L | error L | measured R | predicted R | error R")
    for r in rows:
        print(f"{r['position_mm']:8.1f} | {r['measured_left_mm']:10.2f} | {r['predicted_left_mm']:11.2f} | "
              f"{r['left_error_mm']:+7.2f} | {r['measured_right_mm']:10.2f} | {r['predicted_right_mm']:11.2f} | "
              f"{r['right_error_mm']:+7.2f}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved reusable mirror calibration: {os.path.join(args.out_dir, 'mirror_homographies.json')}")
    print(f"Saved rectified image pairs in: {args.out_dir}")
    if axis_prediction is None:
        print("WARNING: reflection_model_2d_hinge_reference.py not found; predictions are NaN.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
