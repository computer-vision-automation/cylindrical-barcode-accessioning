
"""
phase6_cylinder_mapping.py
==========================

Phase 6: Convert mirror-plane barcode measurements into possible
cylindrical angular positions using reflection_model_2d.py.

Known setup:
    Mirror included angle:       99 degrees
    Camera-to-hinge distance:   240 mm
    Camera on angle bisector:   (0, 240) mm
    Hinge-to-tube-centre:        70 mm
    Tube outer diameter:        13 mm
    Tube radius:                 6.5 mm

Input:
    Phase 5 V2 *_detections.json

Output:
    phase6_mapping.json

IMPORTANT:
    The solver can return multiple possible angular positions.
    This script preserves all returned solutions.
    It does not assume that the first solution is correct.

    The model is a 2D top-down reflection model. It does not yet
    calculate the tube's vertical/axial coordinate mapping.
"""

import argparse
import importlib.util
import json
import os
import math

import numpy as np


# ---------------------------------------------------------------
# Default physical setup
# ---------------------------------------------------------------

DEFAULT_ANGLE_DEG = 99.0
DEFAULT_CAMERA_DISTANCE_MM = 240.0
DEFAULT_TUBE_DISTANCE_MM = 70.0
DEFAULT_TUBE_DIAMETER_MM = 13.0
DEFAULT_GAP_MM = 9.5
DEFAULT_MIRROR_WIDTH_MM = 100.0

MODEL_FILE = "reflection_model_2d.py"


# ---------------------------------------------------------------
# Load reflection model
# ---------------------------------------------------------------

def load_reflection_model(path):
    path = os.path.abspath(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Reflection model not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "reflection_model_2d",
        path
    )

    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    if not hasattr(model, "solve_phi"):
        raise AttributeError(
            "reflection_model_2d.py does not contain solve_phi()."
        )

    return model


# ---------------------------------------------------------------
# Convert model results into JSON-compatible values
# ---------------------------------------------------------------

def make_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


# ---------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------

def build_camera_position(distance_mm, camera_x_mm=0.0):
    """
    Camera lies on the mirror angle bisector.

    Coordinate convention used by the reflection model:
        X = lateral direction
        Y = direction along the angle bisector

    Camera position is relative to the theoretical hinge.
    """

    return [
        float(camera_x_mm),
        float(distance_mm)
    ]


# ---------------------------------------------------------------
# Solve one barcode boundary
# ---------------------------------------------------------------

def solve_boundary(
    model,
    measured_distance_mm,
    side,
    tube_distance_mm,
    camera,
    angle_deg,
    gap_mm,
    mirror_width_mm,
    radius_mm,
    tolerance_mm,
    flask_x_mm
):
    """
    Find all possible cylinder angles for one measured mirror
    boundary.

    solve_phi() scans possible angular positions and returns
    positions whose predicted mirror distance is within tolerance.
    """

    result = model.solve_phi(
        measured_distance_mm,
        side,
        tube_distance_mm,
        camera,
        tol_mm=tolerance_mm,
        angle_deg=angle_deg,
        gap_mm=gap_mm,
        width_mm=mirror_width_mm,
        radius_mm=radius_mm,
        flask_x_mm=flask_x_mm
    )

    return make_json_safe(result)


# ---------------------------------------------------------------
# Process detections
# ---------------------------------------------------------------

def process_detection(
    model,
    detection,
    camera,
    args
):
    side = detection.get("view")

    if side not in ("left", "right"):
        return None

    hinge = detection.get("hinge_edge_distance_mm")

    if not hinge:
        print(
            f"Skipping {side} detection: "
            "hinge-edge measurements missing."
        )
        return None

    near_mm = float(hinge["near"])
    far_mm = float(hinge["far"])

    # Preserve the original mirror measurements.
    output = {
        "view": side,
        "score": detection.get("score"),
        "orientation": detection.get("orientation"),
        "mirror_measurements_mm": {
            "near": near_mm,
            "far": far_mm,
            "centre": float(hinge["centre"])
        },
        "vertical_range_mm_from_top": detection.get(
            "height_range_mm_from_top"
        ),
        "angular_solutions_deg": {}
    }

    # Convert both boundaries independently.
    for boundary_name, measured_distance in (
        ("near", near_mm),
        ("far", far_mm)
    ):
        print(
            f"Solving {side} mirror {boundary_name} boundary: "
            f"{measured_distance:.2f} mm"
        )

        try:
            solutions = solve_boundary(
                model=model,
                measured_distance_mm=measured_distance,
                side=side,
                tube_distance_mm=args.tube_distance,
                camera=camera,
                angle_deg=args.angle,
                gap_mm=args.gap,
                mirror_width_mm=args.mirror_width,
                radius_mm=args.radius,
                tolerance_mm=args.tolerance,
                flask_x_mm=args.flask_x
            )

            output["angular_solutions_deg"][boundary_name] = solutions

        except Exception as exc:
            output["angular_solutions_deg"][boundary_name] = {
                "error": str(exc)
            }

    return output


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def run(args):
    with open(args.input, "r", encoding="utf-8") as f:
        detection_data = json.load(f)

    detections = detection_data.get("detections", [])

    if not detections:
        raise ValueError(
            "No detections found in the input JSON."
        )

    model_path = args.model

    if not os.path.isabs(model_path):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            model_path
        )

    model = load_reflection_model(model_path)

    camera = build_camera_position(
        args.camera_distance,
        args.camera_x
    )

    radius = args.radius

    print("\nPHASE 6: CYLINDER ANGULAR MAPPING")
    print("=" * 55)

    print(f"Mirror angle:             {args.angle:.2f} degrees")
    print(f"Camera position:          {camera} mm")
    print(f"Tube centre distance:     {args.tube_distance:.2f} mm")
    print(f"Tube radius:              {radius:.2f} mm")
    print(f"Mirror gap:               {args.gap:.2f} mm")
    print(f"Mirror width:             {args.mirror_width:.2f} mm")
    print(f"Solver tolerance:         {args.tolerance:.2f} mm")

    print("=" * 55)

    mapped = []

    for detection in detections:
        result = process_detection(
            model,
            detection,
            camera,
            args
        )

        if result is not None:
            mapped.append(result)

    output = {
        "phase": 6,
        "description": (
            "Mirror-plane barcode boundaries converted into "
            "possible cylindrical angular positions."
        ),
        "input_detection_file": args.input,

        "geometry": {
            "mirror_angle_deg": args.angle,
            "camera_position_mm": camera,
            "camera_distance_from_hinge_mm": args.camera_distance,
            "tube_centre_distance_from_hinge_mm": args.tube_distance,
            "tube_diameter_mm": radius * 2.0,
            "tube_radius_mm": radius,
            "mirror_gap_mm": args.gap,
            "mirror_width_mm": args.mirror_width,
            "flask_x_mm": args.flask_x,
            "solver_tolerance_mm": args.tolerance
        },

        "coordinate_notes": {
            "mirror_measurements": (
                "Distances from the hinge-side glass edge."
            ),
            "angular_solutions": (
                "Possible angular positions returned by solve_phi. "
                "Multiple solutions are preserved."
            ),
            "vertical_coordinates": (
                "The original mirror vertical measurements are "
                "preserved but are not converted into tube axial "
                "coordinates by this 2D model."
            ),
            "validation": (
                "Geometric predictions require physical validation "
                "before being used for final barcode reconstruction."
            )
        },

        "mapped_detections": mapped
    }

    output_path = args.output

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nPHASE 6 RESULTS")
    print("=" * 55)

    for i, item in enumerate(mapped):
        print(
            f"\nDetection {i}: {item['view']} mirror"
        )

        for boundary, solutions in item[
            "angular_solutions_deg"
        ].items():
            print(f"  {boundary}: {solutions}")

    print("\nSaved mapping to:", output_path)


# ---------------------------------------------------------------
# Command line
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: mirror-to-cylinder angular mapping"
    )

    parser.add_argument(
        "input",
        help="Phase 5 V2 detection JSON"
    )

    parser.add_argument(
        "--model",
        default=MODEL_FILE,
        help="Path to reflection_model_2d.py"
    )

    parser.add_argument(
        "--output",
        default="phase6_mapping.json"
    )

    parser.add_argument(
        "--angle",
        type=float,
        default=DEFAULT_ANGLE_DEG
    )

    parser.add_argument(
        "--camera-distance",
        type=float,
        default=DEFAULT_CAMERA_DISTANCE_MM
    )

    parser.add_argument(
        "--camera-x",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--tube-distance",
        type=float,
        default=DEFAULT_TUBE_DISTANCE_MM
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_TUBE_DIAMETER_MM / 2.0
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_GAP_MM
    )

    parser.add_argument(
        "--mirror-width",
        type=float,
        default=DEFAULT_MIRROR_WIDTH_MM
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.5
    )

    parser.add_argument(
        "--flask-x",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
