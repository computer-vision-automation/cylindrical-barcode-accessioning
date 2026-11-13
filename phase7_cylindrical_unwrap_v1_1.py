#!/usr/bin/env python3
"""Phase 7: map a Phase-5 mirror crop onto an unwrapped cylindrical surface.

Inputs:
  * Phase-6 V2 JSON (selected angular interval + geometry)
  * Phase-5 detection JSON (crop filename + mirror-plane bounding box)
  * Phase-5 rectified crop image
  * Vertical calibration JSON containing >=2 (mirror_y_mm, tube_z_mm) anchors
  * reflection_model_2d.py used by Phase 6 V2

The vertical calibration maps rectified mirror height to tube axial height.
The angular mapping is recomputed from the same Phase-6 reflection model.
No crop rotation is applied. The mirror's lateral reflection is accounted for
by the nonlinear mirror-distance-to-angle remap.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys

import cv2
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_model(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reflection model not found: {path}")
    spec = importlib.util.spec_from_file_location("reflection_model_2d", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reflection model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise AttributeError("Reflection model must provide predict().")
    return module


def write_calibration_template(path):
    template = {
        "description": "Map rectified mirror vertical coordinates to tube axial coordinates.",
        "coordinate_definition": {
            "mirror_y_mm": "Distance downward from the top of the rectified mirror plane.",
            "tube_z_mm": "Distance downward from the chosen physical tube-top reference.",
        },
        "anchors": [
            {"mirror_y_mm": None, "tube_z_mm": None, "label": "known_reference_1"},
            {"mirror_y_mm": None, "tube_z_mm": None, "label": "known_reference_2"},
        ],
        "notes": [
            "Fill both anchor pairs using known physical reference positions.",
            "Use distinct mirror_y_mm values. More than two measured anchors are supported.",
            "Do not enter the barcode's detected top/bottom as calibration anchors unless their physical tube heights are independently known.",
            "For a vertical mirror and vertical tube, the expected scale is near +1; a large deviation suggests calibration or alignment error.",
        ],
    }
    save_json(path, template)


def fit_vertical_calibration(path):
    data = load_json(path)
    anchors = data.get("anchors", [])
    points = []
    for i, item in enumerate(anchors):
        y = item.get("mirror_y_mm")
        z = item.get("tube_z_mm")
        if y is None or z is None:
            continue
        y, z = float(y), float(z)
        if not (math.isfinite(y) and math.isfinite(z)):
            raise ValueError(f"Calibration anchor {i + 1} contains a non-finite value.")
        points.append((y, z))
    if len(points) < 2:
        raise ValueError("Vertical calibration needs at least two complete anchor pairs.")

    ys = np.asarray([p[0] for p in points], dtype=np.float64)
    zs = np.asarray([p[1] for p in points], dtype=np.float64)
    if np.ptp(ys) < 1e-6:
        raise ValueError("Calibration anchors must have different mirror_y_mm values.")
    slope, intercept = np.polyfit(ys, zs, 1)
    predicted = slope * ys + intercept
    residuals = zs - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    if slope <= 0:
        raise ValueError(
            f"Fitted vertical scale is {slope:.6f}; expected a positive scale because both coordinates increase downward. Check anchor definitions."
        )
    return {
        "scale_tube_mm_per_mirror_mm": float(slope),
        "offset_tube_mm": float(intercept),
        "anchor_count": len(points),
        "anchor_rmse_mm": rmse,
        "anchors_used": [
            {"mirror_y_mm": float(y), "tube_z_mm": float(z)} for y, z in points
        ],
    }


def get_detection(phase5, phase6_detection):
    detections = phase5.get("detections", [])
    view = phase6_detection.get("view")
    candidates = [d for d in detections if d.get("view") == view]
    if not candidates:
        raise ValueError(f"No Phase-5 detection matches Phase-6 view '{view}'.")
    # Prefer the candidate whose mirror boundaries match the Phase-6 record.
    p6m = phase6_detection.get("mirror_measurements_mm", {})
    target_near = float(p6m.get("near"))
    target_far = float(p6m.get("far"))
    def mismatch(d):
        m = d.get("hinge_edge_distance_mm", {})
        return abs(float(m.get("near", 1e9)) - target_near) + abs(float(m.get("far", 1e9)) - target_far)
    return min(candidates, key=mismatch)


def resolve_crop_path(phase5_path, detection, explicit_crop=None):
    if explicit_crop:
        path = explicit_crop
    else:
        crop_name = detection.get("crop_file")
        if not crop_name:
            raise ValueError("Phase-5 detection has no crop_file. Supply --crop explicitly.")
        path = crop_name
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(phase5_path)), path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Barcode crop not found: {path}. Supply its path with --crop.")
    return os.path.abspath(path)


def predict_distance(model, phi_deg, side, geom):
    camera = geom.get("camera_position_mm", [0.0, float(geom.get("camera_distance_from_hinge_mm", 240.0))])
    result = model.predict(
        float(geom["tube_centre_distance_from_hinge_mm"]),
        float(phi_deg),
        camera,
        angle_deg=float(geom["mirror_angle_deg"]),
        gap_mm=float(geom["mirror_gap_mm"]),
        width_mm=float(geom["mirror_width_mm"]),
        radius_mm=float(geom["tube_radius_mm"]),
        flask_x_mm=float(geom.get("flask_x_mm", 0.0)),
    )[side]
    if result is None or not result.get("on_glass") or not result.get("facing"):
        raise ValueError(f"Model says angle {phi_deg:.4f} deg is not valid on {side} mirror.")
    return float(result["d_edge"])


def build_angle_distance_curve(model, side, geom, interval, samples=4001):
    start = float(interval["angle_start_deg"])
    end = float(interval["angle_end_deg"])
    if end <= start:
        raise ValueError("Selected angular interval must have increasing angle bounds.")
    phis = np.linspace(start, end, samples, dtype=np.float64)
    distances = np.asarray([predict_distance(model, p, side, geom) for p in phis])
    diffs = np.diff(distances)
    significant = diffs[np.abs(diffs) > 1e-5]
    if significant.size:
        pos = np.count_nonzero(significant > 0)
        neg = np.count_nonzero(significant < 0)
        minority = min(pos, neg) / significant.size
        if minority > 0.02:
            raise ValueError("Mirror-distance-to-angle curve is not monotonic enough for a stable unwrap.")
    # np.interp requires ascending x values.
    if distances[0] > distances[-1]:
        d_sorted = distances[::-1]
        phi_sorted = phis[::-1]
    else:
        d_sorted = distances
        phi_sorted = phis
    if np.any(np.diff(d_sorted) <= 0):
        raise ValueError("Mirror-distance samples contain duplicate/non-increasing values.")
    return phis, distances, d_sorted, phi_sorted


def run(args):
    phase6_path = os.path.abspath(args.phase6_json)
    phase6 = load_json(phase6_path)
    if phase6.get("phase") != 6 or str(phase6.get("version")) not in ("2.0", "2"):
        raise ValueError("Expected a Phase 6 V2 JSON file.")

    phase5_path = args.phase5_json
    if not phase5_path:
        candidate = phase6.get("input_detection_file", "")
        if candidate:
            candidate = candidate.replace("\\", os.sep)
            if not os.path.isabs(candidate):
                candidate = os.path.join(os.path.dirname(phase6_path), candidate)
            if os.path.isfile(candidate):
                phase5_path = candidate
    if not phase5_path:
        raise ValueError("Could not locate Phase-5 JSON. Supply it with --phase5-json.")
    phase5_path = os.path.abspath(phase5_path)
    phase5 = load_json(phase5_path)

    # Physical axial calibration is optional. Without it, preserve the crop's
    # rectified mirror-plane Y coordinate and label it explicitly as uncalibrated.
    use_axial_calibration = bool(args.axial_calibration)
    if use_axial_calibration:
        if not os.path.isfile(args.axial_calibration):
            write_calibration_template(args.axial_calibration)
            raise FileNotFoundError(
                f"Created calibration template at '{args.axial_calibration}'. Fill in at least two known anchor pairs, then rerun Phase 7."
            )
        vertical = fit_vertical_calibration(args.axial_calibration)
    else:
        vertical = {
            "scale_tube_mm_per_mirror_mm": 1.0,
            "offset_tube_mm": 0.0,
            "anchor_count": 0,
            "anchor_rmse_mm": None,
            "anchors_used": [],
            "calibration_status": "not_calibrated_mirror_plane_y_preserved",
        }

    mapped = phase6.get("mapped_detections", [])
    if not mapped:
        raise ValueError("Phase 6 JSON contains no mapped_detections.")
    selected = [d for d in mapped if d.get("paired_angular_interval", {}).get("status") == "unique_valid_pair"]
    if args.detection_index < 0 or args.detection_index >= len(mapped):
        raise IndexError(f"--detection-index must be between 0 and {len(mapped)-1}.")
    p6d = mapped[args.detection_index]
    pair = p6d.get("paired_angular_interval", {})
    interval = pair.get("selected_interval")
    if pair.get("status") != "unique_valid_pair" or not interval:
        raise ValueError("Selected detection does not have a unique_valid_pair in Phase 6.")

    detection = get_detection(phase5, p6d)
    crop_path = resolve_crop_path(phase5_path, detection, args.crop)
    crop = cv2.imread(crop_path, cv2.IMREAD_COLOR)
    if crop is None:
        raise ValueError(f"OpenCV could not read crop image: {crop_path}")
    crop_h, crop_w = crop.shape[:2]

    box_mm = detection.get("box_rectified_mm")
    if not box_mm or len(box_mm) != 4:
        raise ValueError("Phase-5 detection is missing box_rectified_mm [x,y,width,height].")
    mirror_x0, mirror_y0, mirror_w, mirror_h = map(float, box_mm)
    mirror_x1 = mirror_x0 + mirror_w
    mirror_y1 = mirror_y0 + mirror_h
    measured = p6d.get("mirror_measurements_mm", {})
    near_mm = float(measured["near"])
    far_mm = float(measured["far"])
    if abs((mirror_x0 - near_mm)) > 1.0 or abs((mirror_x1 - far_mm)) > 1.0:
        print("WARNING: Phase-5 crop x-range and Phase-6 near/far measurements differ by more than 1 mm.", file=sys.stderr)

    geom = phase6["geometry"]
    model = load_model(args.model)
    phis, distances, d_sorted, phi_sorted = build_angle_distance_curve(
        model, p6d["view"], geom, interval, samples=args.curve_samples
    )

    # Output x increases with cylinder angle. Since mirror distance decreases
    # with angle in this case, this remap naturally reverses the mirror crop
    # horizontally without rotating the barcode's vertical sequence axis.
    radius = float(geom["tube_radius_mm"])
    theta_start = math.radians(float(interval["angle_start_deg"]))
    theta_end = math.radians(float(interval["angle_end_deg"]))
    arc_mm = radius * (theta_end - theta_start)
    if arc_mm <= 0:
        raise ValueError("Computed surface arc length is not positive.")

    # Use crop box dimensions as mirror-plane coordinate extents.
    output_width = max(2, int(round(arc_mm * args.px_per_mm)))
    y0_tube = vertical["scale_tube_mm_per_mirror_mm"] * mirror_y0 + vertical["offset_tube_mm"]
    y1_tube = vertical["scale_tube_mm_per_mirror_mm"] * mirror_y1 + vertical["offset_tube_mm"]
    z_min, z_max = sorted((y0_tube, y1_tube))
    axial_height = z_max - z_min
    if axial_height <= 0:
        raise ValueError("Mapped vertical height is not positive.")
    output_height = max(2, int(round(axial_height * args.px_per_mm)))

    theta_out = np.linspace(float(interval["angle_start_deg"]), float(interval["angle_end_deg"]), output_width)
    d_out = np.interp(theta_out, phis, distances)
    # Crop x coordinate corresponds to mirror distance increasing left-to-right.
    src_x = (d_out - mirror_x0) / mirror_w * (crop_w - 1)

    z_out = np.linspace(z_min, z_max, output_height)
    mirror_y_out = (z_out - vertical["offset_tube_mm"]) / vertical["scale_tube_mm_per_mirror_mm"]
    src_y = (mirror_y_out - mirror_y0) / mirror_h * (crop_h - 1)

    map_x, map_y = np.meshgrid(src_x.astype(np.float32), src_y.astype(np.float32))
    unwrapped = cv2.remap(
        crop, map_x, map_y, interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    output_image = os.path.abspath(args.output_image)
    os.makedirs(os.path.dirname(output_image) or ".", exist_ok=True)
    if not cv2.imwrite(output_image, unwrapped):
        raise IOError(f"Could not write unwrapped image: {output_image}")

    # Emit a compact, per-column coordinate lookup for audit/reproducibility.
    csv_path = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["output_x_px", "surface_s_mm", "theta_deg", "mirror_distance_mm", "source_crop_x_px"])
        for i, (theta, d, sx) in enumerate(zip(theta_out, d_out, src_x)):
            writer.writerow([i, (i / max(1, output_width - 1)) * arc_mm, float(theta), float(d), float(sx)])

    result = {
        "phase": 7,
        "version": "1.0",
        "description": "Phase-5 rectified mirror crop unwrapped onto the cylindrical surface using Phase-6 angular geometry and measured vertical calibration.",
        "inputs": {
            "phase6_json": phase6_path,
            "phase5_json": phase5_path,
            "crop_image": crop_path,
            "vertical_calibration": (os.path.abspath(args.axial_calibration) if use_axial_calibration else None),
            "reflection_model": os.path.abspath(args.model),
        },
        "detection": {
            "index": args.detection_index,
            "view": p6d["view"],
            "orientation": p6d.get("orientation"),
            "score": p6d.get("score"),
            "phase5_box_rectified_mm": box_mm,
            "phase5_vertical_range_mm_from_mirror_top": [mirror_y0, mirror_y1],
        },
        "angular_mapping": {
            "start_deg": float(interval["angle_start_deg"]),
            "end_deg": float(interval["angle_end_deg"]),
            "width_deg": float(interval["angular_width_deg"]),
            "surface_arc_length_mm": float(arc_mm),
            "mapping_samples": int(args.curve_samples),
            "distance_mapping_direction": interval.get("distance_mapping_direction"),
        },
        "axial_mapping": {
            **vertical,
            "tube_z_range_mm": ([float(z_min), float(z_max)] if use_axial_calibration else None),
            "axial_height_mm": (float(axial_height) if use_axial_calibration else None),
            "mirror_plane_y_range_mm": [float(mirror_y0), float(mirror_y1)],
            "mirror_plane_height_mm": float(mirror_h),
            "calibration_rmse_mm": vertical["anchor_rmse_mm"],
            "note": (
                "Physical tube axial coordinates derived from supplied calibration."
                if use_axial_calibration else
                "No physical axial calibration supplied. Vertical crop coordinate is preserved; these values are mirror-plane coordinates, not tube axial measurements."
            ),
        },
        "output": {
            "image": output_image,
            "coordinate_csv": csv_path,
            "width_px": int(output_width),
            "height_px": int(output_height),
            "pixels_per_mm": float(args.px_per_mm),
            "orientation_note": "Barcode sequence axis remains vertical; no 90-degree rotation or transpose was applied.",
            "pixel_mapping_note": "The horizontal remap is nonlinear and derived from the Phase-6 reflection model. The vertical remap uses the supplied affine calibration anchors.",
        },
        "validation_notes": [
            "The angular geometry is model-derived and still requires physical validation.",
            "Without vertical calibration, the output preserves mirror-plane Y and must not be interpreted as physical tube axial coordinates.",
            "With vertical calibration, axial accuracy depends on supplied anchors and rectified crop calibration.",
            "This script unwraps the detected mirror crop; it does not combine direct and both mirror views.",
            "Check the output image visually and run barcode decoding separately before treating it as a successful read.",
        ],
    }
    save_json(os.path.abspath(args.output_json), result)

    print("PHASE 7: CYLINDRICAL BARCODE UNWRAPPING")
    print("=" * 62)
    print(f"View:                       {p6d['view']}")
    print(f"Input crop:                 {crop_w} x {crop_h} px")
    print(f"Angular range:              {interval['angle_start_deg']:.4f} to {interval['angle_end_deg']:.4f} deg")
    print(f"Surface arc:                {arc_mm:.4f} mm")
    if use_axial_calibration:
        print(f"Vertical calibration scale: {vertical['scale_tube_mm_per_mirror_mm']:.6f}")
        print(f"Vertical calibration RMSE:  {vertical['anchor_rmse_mm']:.4f} mm")
        print(f"Tube axial range:           {z_min:.3f} to {z_max:.3f} mm")
    else:
        print("Vertical mapping:           uncalibrated; mirror-plane Y preserved")
        print(f"Mirror-plane Y range:       {mirror_y0:.3f} to {mirror_y1:.3f} mm")
    print(f"Output image:               {output_width} x {output_height} px")
    print(f"Saved image:                {output_image}")
    print(f"Saved coordinate CSV:       {csv_path}")
    print(f"Saved mapping JSON:         {os.path.abspath(args.output_json)}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7 cylindrical unwrapping of a Phase-5 mirror crop.")
    parser.add_argument("phase6_json", help="Phase 6 V2 mapping JSON")
    parser.add_argument("--phase5-json", help="Phase 5 detection JSON; auto-detected from Phase 6 when possible")
    parser.add_argument("--crop", help="Explicit path to the Phase-5 rectified barcode crop image")
    parser.add_argument("--axial-calibration", default=None, help="Optional JSON with >=2 mirror-y to tube-z anchor pairs; omit to preserve uncalibrated mirror-plane Y")
    parser.add_argument("--model", default="reflection_model_2d.py", help="Same reflection_model_2d.py used by Phase 6 V2")
    parser.add_argument("--detection-index", type=int, default=0)
    parser.add_argument("--px-per-mm", type=float, default=10.0)
    parser.add_argument("--curve-samples", type=int, default=4001)
    parser.add_argument("--output-image", default="phase7_unwrapped_barcode.png")
    parser.add_argument("--output-csv", default="phase7_coordinate_map.csv")
    parser.add_argument("--output-json", default="phase7_mapping.json")
    parser.add_argument("--write-calibration-template", metavar="PATH", help="Write an empty vertical-calibration template and exit")
    args = parser.parse_args()

    if args.px_per_mm <= 0:
        parser.error("--px-per-mm must be positive")
    if args.curve_samples < 101:
        parser.error("--curve-samples must be at least 101")
    if args.write_calibration_template:
        write_calibration_template(args.write_calibration_template)
        print(f"Wrote vertical calibration template: {args.write_calibration_template}")
        return
    try:
        run(args)
    except Exception as exc:
        print(f"PHASE 7 ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
