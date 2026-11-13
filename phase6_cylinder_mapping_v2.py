"""Phase 6 V2: map Phase 5 mirror-plane barcode bounds to cylinder angles.

This version keeps the Phase 5 input JSON unchanged and preserves the original
Phase 6 output fields where practical. It distinguishes:
  * exact interpolated boundary crossings;
  * tolerance-accepted angular ranges;
  * physically checked near/far endpoint pairings.

The reflection model is still a top-down 2D prototype. It does NOT map the
vertical/axial coordinate of the barcode onto the tube.
"""

import argparse
import importlib.util
import json
import math
import os

import numpy as np

DEFAULT_ANGLE_DEG = 99.0
DEFAULT_CAMERA_DISTANCE_MM = 240.0
DEFAULT_TUBE_DISTANCE_MM = 70.0
DEFAULT_TUBE_DIAMETER_MM = 13.0
DEFAULT_GAP_MM = 9.5
DEFAULT_MIRROR_WIDTH_MM = 100.0
DEFAULT_TOLERANCE_MM = 1.5
DEFAULT_ANGLE_STEP_DEG = 0.1
MODEL_FILE = "reflection_model_2d.py"


def load_reflection_model(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reflection model not found: {path}")
    spec = importlib.util.spec_from_file_location("reflection_model_2d", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reflection model: {path}")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    if not hasattr(model, "predict"):
        raise AttributeError("reflection_model_2d.py does not contain predict().")
    return model


def make_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_camera_position(distance_mm, camera_x_mm=0.0):
    return [float(camera_x_mm), float(distance_mm)]


def predict_edge(model, phi_deg, side, args, camera):
    """Return the model prediction and validity flags at one cylinder angle."""
    result = model.predict(
        args.tube_distance,
        float(phi_deg),
        camera,
        angle_deg=args.angle,
        gap_mm=args.gap,
        width_mm=args.mirror_width,
        radius_mm=args.radius,
        flask_x_mm=args.flask_x,
    )[side]
    if result is None:
        return None
    return {
        "d_edge_mm": float(result["d_edge"]),
        "on_glass": bool(result["on_glass"]),
        "facing": bool(result["facing"]),
    }


def contiguous_ranges(values, max_gap_deg):
    """Group sorted angular samples into contiguous ranges."""
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= max_gap_deg:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def analyze_boundary(model, side, measured_mm, args, camera):
    """Find exact crossings and tolerance bands for one mirror boundary."""
    step = args.angle_step
    # Include +180 so the complete requested angular domain is sampled.
    angles = np.arange(-180.0, 180.0 + step * 0.5, step)
    predictions = [predict_edge(model, a, side, args, camera) for a in angles]

    residuals = []
    valid = []
    for pred in predictions:
        is_valid = bool(pred and pred["on_glass"] and pred["facing"])
        valid.append(is_valid)
        residuals.append(
            pred["d_edge_mm"] - measured_mm if is_valid else np.nan
        )

    # Exact crossings are linearly interpolated between adjacent valid samples.
    roots = []
    for i in range(len(angles) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        e0, e1 = residuals[i], residuals[i + 1]
        if abs(e0) < 1e-10:
            roots.append(float(angles[i]))
        if e0 * e1 < 0.0:
            fraction = -e0 / (e1 - e0)
            root = float(angles[i] + fraction * (angles[i + 1] - angles[i]))
            roots.append(root)
    if valid[-1] and abs(residuals[-1]) < 1e-10:
        roots.append(float(angles[-1]))

    roots.sort()
    deduped_roots = []
    for root in roots:
        if not deduped_roots or abs(root - deduped_roots[-1]) > step * 0.75:
            pred = predict_edge(model, root, side, args, camera)
            deduped_roots.append({
                "angle_deg": round(root, 4),
                "predicted_distance_mm": round(pred["d_edge_mm"], 4),
                "absolute_error_mm": round(abs(pred["d_edge_mm"] - measured_mm), 6),
            })

    # Tolerance samples, grouped into ranges. These are not exact crossings.
    accepted = [
        float(angles[i]) for i in range(len(angles))
        if valid[i] and abs(residuals[i]) <= args.tolerance
    ]
    groups = contiguous_ranges(accepted, step * 1.51)
    tolerance_ranges = []
    for group in groups:
        best_angle = min(
            group,
            key=lambda a: abs(predict_edge(model, a, side, args, camera)["d_edge_mm"] - measured_mm),
        )
        best_pred = predict_edge(model, best_angle, side, args, camera)
        tolerance_ranges.append({
            "start_deg": round(group[0], 3),
            "end_deg": round(group[-1], 3),
            "best_sample_angle_deg": round(best_angle, 3),
            "best_sample_error_mm": round(abs(best_pred["d_edge_mm"] - measured_mm), 5),
        })

    # Best valid sample is useful even when no exact crossing exists.
    valid_indices = [i for i, ok in enumerate(valid) if ok]
    best_sample = None
    if valid_indices:
        best_i = min(valid_indices, key=lambda i: abs(residuals[i]))
        best_sample = {
            "angle_deg": round(float(angles[best_i]), 3),
            "predicted_distance_mm": round(float(predictions[best_i]["d_edge_mm"]), 4),
            "absolute_error_mm": round(abs(float(residuals[best_i])), 5),
        }

    # Keep a numeric list for compatibility with the original Phase 6 output.
    legacy_tolerance_angles = [round(a, 3) for a in accepted]
    return {
        "measured_distance_mm": float(measured_mm),
        "exact_crossings": deduped_roots,
        "tolerance_ranges": tolerance_ranges,
        "best_valid_sample": best_sample,
        "tolerance_mm": float(args.tolerance),
        "angular_solutions_deg": legacy_tolerance_angles,
    }


def check_pair(model, side, near_angle, far_angle, near_mm, far_mm, args, camera):
    """Check whether the interval between two exact roots is continuously valid."""
    lo, hi = sorted((float(near_angle), float(far_angle)))
    if hi - lo < args.angle_step:
        return {"valid": False, "reason": "Angular interval is effectively zero."}

    sample_angles = np.linspace(lo, hi, max(101, int((hi - lo) / 0.25) + 1))
    distances = []
    for angle in sample_angles:
        pred = predict_edge(model, angle, side, args, camera)
        if pred is None or not pred["on_glass"] or not pred["facing"]:
            return {
                "valid": False,
                "reason": "At least one intermediate angle is not visible/facing/on the mirror glass.",
            }
        distances.append(pred["d_edge_mm"])

    low_d, high_d = sorted((near_mm, far_mm))
    margin = max(0.05, args.pair_margin_mm)
    if min(distances) < low_d - margin or max(distances) > high_d + margin:
        return {
            "valid": False,
            "reason": "Intermediate predictions leave the measured boundary interval.",
        }

    diffs = np.diff(distances)
    significant = diffs[np.abs(diffs) > 0.002]
    monotonic = True
    direction = "approximately constant"
    if significant.size:
        positive = np.count_nonzero(significant > 0)
        negative = np.count_nonzero(significant < 0)
        minority_fraction = min(positive, negative) / significant.size
        monotonic = minority_fraction <= 0.05
        direction = "increasing" if positive >= negative else "decreasing"
    if not monotonic:
        return {
            "valid": False,
            "reason": "Predicted mirror distance is not sufficiently monotonic across the interval.",
        }

    angular_width = hi - lo
    arc_length = args.radius * math.radians(angular_width)
    return {
        "valid": True,
        "angle_start_deg": round(lo, 4),
        "angle_end_deg": round(hi, 4),
        "angular_width_deg": round(angular_width, 4),
        "surface_arc_length_mm": round(arc_length, 4),
        "predicted_distance_min_mm": round(float(min(distances)), 4),
        "predicted_distance_max_mm": round(float(max(distances)), 4),
        "distance_mapping_direction": direction,
        "intermediate_samples_checked": int(len(sample_angles)),
    }


def pair_boundaries(model, side, near_result, far_result, near_mm, far_mm, args, camera):
    near_roots = near_result["exact_crossings"]
    far_roots = far_result["exact_crossings"]
    tested = []
    for nr in near_roots:
        for fr in far_roots:
            check = check_pair(
                model, side, nr["angle_deg"], fr["angle_deg"],
                near_mm, far_mm, args, camera,
            )
            tested.append({
                "near_boundary_angle_deg": nr["angle_deg"],
                "far_boundary_angle_deg": fr["angle_deg"],
                **check,
            })

    valid_pairs = [p for p in tested if p.get("valid")]
    if len(valid_pairs) == 1:
        status = "unique_valid_pair"
        selected = valid_pairs[0]
    elif len(valid_pairs) > 1:
        status = "ambiguous_multiple_valid_pairs"
        selected = None
    elif not near_roots or not far_roots:
        status = "no_exact_crossing_for_one_or_both_boundaries"
        selected = None
    else:
        status = "no_pair_passed_continuity_checks"
        selected = None

    return {
        "status": status,
        "selected_interval": selected,
        "tested_pairs": tested,
        "note": (
            "A selected interval is a model-derived candidate, not a physically validated result."
        ),
    }


def process_detection(model, detection, args, camera):
    side = detection.get("view")
    if side not in ("left", "right"):
        return None

    hinge = detection.get("hinge_edge_distance_mm")
    if not isinstance(hinge, dict) or "near" not in hinge or "far" not in hinge:
        print(f"Skipping {side} detection: hinge-edge measurements missing.")
        return None

    near_mm = float(hinge["near"])
    far_mm = float(hinge["far"])
    print(f"Analysing {side} mirror boundaries: near={near_mm:.2f} mm, far={far_mm:.2f} mm")

    near_result = analyze_boundary(model, side, near_mm, args, camera)
    far_result = analyze_boundary(model, side, far_mm, args, camera)
    paired = pair_boundaries(
        model, side, near_result, far_result, near_mm, far_mm, args, camera
    )

    return {
        "view": side,
        "score": detection.get("score"),
        "orientation": detection.get("orientation"),
        "mirror_measurements_mm": {
            "near": near_mm,
            "far": far_mm,
            "centre": float(hinge.get("centre", (near_mm + far_mm) / 2.0)),
        },
        "vertical_range_mm_from_top": detection.get("height_range_mm_from_top"),
        # Compatibility field: tolerance-accepted samples, not exact roots.
        "angular_solutions_deg": {
            "near": near_result["angular_solutions_deg"],
            "far": far_result["angular_solutions_deg"],
        },
        "boundary_analysis": {
            "near": {k: v for k, v in near_result.items() if k != "angular_solutions_deg"},
            "far": {k: v for k, v in far_result.items() if k != "angular_solutions_deg"},
        },
        "paired_angular_interval": paired,
    }


def run(args):
    with open(args.input, "r", encoding="utf-8") as f:
        detection_data = json.load(f)
    detections = detection_data.get("detections", [])
    if not detections:
        raise ValueError("No detections found in the input JSON.")

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    model = load_reflection_model(model_path)
    camera = build_camera_position(args.camera_distance, args.camera_x)

    print("\nPHASE 6 V2: EXACT + TOLERANCE-AWARE CYLINDER MAPPING")
    print("=" * 64)
    print(f"Mirror angle:             {args.angle:.2f} degrees")
    print(f"Camera position:          {camera} mm")
    print(f"Tube centre distance:     {args.tube_distance:.2f} mm")
    print(f"Tube radius:              {args.radius:.2f} mm")
    print(f"Mirror gap:               {args.gap:.2f} mm")
    print(f"Mirror width:             {args.mirror_width:.2f} mm")
    print(f"Solver tolerance:         +/- {args.tolerance:.2f} mm")
    print(f"Angular scan step:        {args.angle_step:.3f} degrees")
    print("=" * 64)

    mapped = []
    for detection in detections:
        result = process_detection(model, detection, args, camera)
        if result is not None:
            mapped.append(result)

    output = {
        "phase": 6,
        "version": "2.0",
        "description": (
            "Phase 5 mirror-plane barcode boundaries mapped to exact angular crossings, "
            "tolerance ranges, and continuity-checked candidate intervals."
        ),
        "input_detection_file": args.input,
        "geometry": {
            "mirror_angle_deg": args.angle,
            "camera_position_mm": camera,
            "camera_distance_from_hinge_mm": args.camera_distance,
            "tube_centre_distance_from_hinge_mm": args.tube_distance,
            "tube_diameter_mm": args.radius * 2.0,
            "tube_radius_mm": args.radius,
            "mirror_gap_mm": args.gap,
            "mirror_width_mm": args.mirror_width,
            "flask_x_mm": args.flask_x,
            "solver_tolerance_mm": args.tolerance,
            "angular_scan_step_deg": args.angle_step,
        },
        "coordinate_notes": {
            "mirror_measurements": "Distances from the hinge-side glass edge on the rectified mirror plane.",
            "exact_crossings": "Linearly interpolated angles where the model prediction equals the measured mirror distance.",
            "tolerance_ranges": "Angles whose valid model predictions are within the configured tolerance; these are uncertainty bands, not exact solutions.",
            "paired_interval": "Selected only if exactly one pair of exact crossings passes visibility, glass-boundary, measured-range, and monotonicity checks.",
            "vertical_coordinates": "Original Phase 5 vertical measurements are preserved but are not converted to tube axial coordinates by this top-down 2D model.",
            "orientation": "Barcode orientation is preserved as supplied by Phase 5. No crop rotation or coordinate transposition is performed.",
            "validation": "All angular results remain model estimates and require physical validation against measured tube/barcode positions.",
        },
        "mapped_detections": mapped,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(output), f, indent=2)

    print("\nPHASE 6 V2 SUMMARY")
    for i, item in enumerate(mapped):
        print(f"\nDetection {i}: {item['view']} mirror")
        for boundary in ("near", "far"):
            analysis = item["boundary_analysis"][boundary]
            roots = analysis["exact_crossings"]
            root_text = ", ".join(f"{r['angle_deg']:.2f} deg" for r in roots) or "none"
            print(f"  {boundary} exact crossing(s): {root_text}")
            print(f"  {boundary} tolerance range(s): {len(analysis['tolerance_ranges'])}")
        pair = item["paired_angular_interval"]
        print(f"  paired interval status: {pair['status']}")
        selected = pair.get("selected_interval")
        if selected:
            print(
                f"  selected interval: {selected['angle_start_deg']:.2f} to "
                f"{selected['angle_end_deg']:.2f} deg "
                f"({selected['angular_width_deg']:.2f} deg; "
                f"arc {selected['surface_arc_length_mm']:.2f} mm)"
            )
    print(f"\nSaved mapping to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 V2: exact and tolerance-aware mirror-to-cylinder mapping"
    )
    parser.add_argument("input", help="Phase 5 V2 detection JSON")
    parser.add_argument("--model", default=MODEL_FILE, help="Path to reflection_model_2d.py")
    parser.add_argument("--output", default="phase6_mapping_v2.json")
    parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE_DEG)
    parser.add_argument("--camera-distance", type=float, default=DEFAULT_CAMERA_DISTANCE_MM)
    parser.add_argument("--camera-x", type=float, default=0.0)
    parser.add_argument("--tube-distance", type=float, default=DEFAULT_TUBE_DISTANCE_MM)
    parser.add_argument("--radius", type=float, default=DEFAULT_TUBE_DIAMETER_MM / 2.0)
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP_MM)
    parser.add_argument("--mirror-width", type=float, default=DEFAULT_MIRROR_WIDTH_MM)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_MM)
    parser.add_argument("--angle-step", type=float, default=DEFAULT_ANGLE_STEP_DEG)
    parser.add_argument("--pair-margin-mm", type=float, default=0.10)
    parser.add_argument("--flask-x", type=float, default=0.0)
    args = parser.parse_args()

    if args.angle_step <= 0 or args.angle_step > 5:
        parser.error("--angle-step must be greater than 0 and no more than 5 degrees")
    if args.tolerance < 0:
        parser.error("--tolerance cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
