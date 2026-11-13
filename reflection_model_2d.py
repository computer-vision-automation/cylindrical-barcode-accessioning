"""
Phase 4 - Top-down 2D reflection model for the two-mirror rig.

Predicts where a flask or barcode point appears along each mirror.
Flask-axis distances are measured from the physical hinge reference,
which this 2D model treats as the theoretical intersection of the
extended mirror planes. The glass inner edges are offset from that
origin by the measured hinge gap.

IMPORTANT:
This is a geometric prototype. Its predictions must be validated
against physical measurements before being treated as accurate.
The physical hinge pivot is assumed to coincide with the theoretical
intersection of the mirror planes. If the hinge hardware places the
pivot elsewhere, measure that offset and update the origin model.
"""

import argparse
import math

import numpy as np


# ------------------------------------------------------------
# MIRROR GEOMETRY
# ------------------------------------------------------------

def mirror_dirs(angle_deg):
    """Unit direction vectors for the left and right mirror planes."""
    a = math.radians(angle_deg) / 2.0

    return {
        "left": np.array([-math.sin(a), math.cos(a)]),
        "right": np.array([math.sin(a), math.cos(a)])
    }


def hinge_offset_s0(angle_deg, gap_mm):
    """Distance along each mirror from the theoretical vertex to its inner edge."""
    return gap_mm / (2.0 * math.sin(math.radians(angle_deg) / 2.0))


def normal_of(u):
    return np.array([-u[1], u[0]])


# ------------------------------------------------------------
# REFLECTION GEOMETRY
# ------------------------------------------------------------

def reflect_point(P, u):
    """Reflect point P across a mirror line through the origin."""
    P = np.asarray(P, dtype=float)
    return 2.0 * np.dot(P, u) * u - P


def reflect_dir(v, u):
    """Reflect direction vector v across mirror line u."""
    n = normal_of(u)
    return v - 2.0 * np.dot(v, n) * n


def surface_point(d_mm, phi_deg, radius_mm, x_mm=0.0):
    """
    Position of a point on the cylindrical flask surface.

    d_mm: flask-axis distance from the hinge reference along the bisector
    phi_deg: angular position around the flask
    radius_mm: flask radius
    x_mm: sideways flask-axis offset
    """
    p = math.radians(phi_deg)

    return np.array([
        x_mm + radius_mm * math.sin(p),
        d_mm + radius_mm * math.cos(p)
    ])


def seen_in_mirror(C, P, u, s0, width_mm, surface_normal=None):
    """
    Calculate where the camera sees point P reflected in a mirror.

    Returns:
        s: distance along mirror from theoretical vertex
        d_edge: distance from hinge-side glass edge
        on_glass: whether the reflection falls within the glass
        facing: whether the surface faces the mirror path
        S: reflection point on mirror
        virtual: virtual image position

    Returns None if no valid reflection exists.
    """

    C = np.asarray(C, dtype=float)
    P = np.asarray(P, dtype=float)

    n = normal_of(u)

    # Camera and object must be on the same side of the mirror.
    if np.dot(C, n) * np.dot(P, n) <= 0:
        return None

    # Construct the virtual image behind the mirror.
    Pv = reflect_point(P, u)

    dC = np.dot(C, n)
    dV = np.dot(Pv, n)

    denominator = dC - dV

    if abs(denominator) < 1e-12:
        return None

    # Intersection of camera-to-virtual-point line with mirror.
    t = dC / denominator

    if not (0.0 <= t <= 1.0):
        return None

    S = C + t * (Pv - C)

    # Position along mirror.
    s = float(np.dot(S, u))

    # Distance from hinge-side glass edge.
    d_edge = s - s0

    on_glass = (s0 <= s <= s0 + width_mm)

    facing = True

    if surface_normal is not None:
        facing = bool(np.dot(surface_normal, S - P) > 0)

    return {
        "s": s,
        "d_edge": d_edge,
        "on_glass": on_glass,
        "facing": facing,
        "S": S,
        "virtual": Pv
    }


def direct_view(C, P, surface_normal):
    """Check whether the barcode surface point faces the camera."""
    return bool(
        np.dot(
            surface_normal,
            np.asarray(C, dtype=float) - np.asarray(P, dtype=float)
        ) > 0
    )


# ------------------------------------------------------------
# PREDICTION FUNCTIONS
# ------------------------------------------------------------

def predict(
    d_mm,
    phi_deg,
    camera,
    angle_deg=99.0,
    gap_mm=9.5,
    width_mm=100.0,
    radius_mm=6.5,
    flask_x_mm=0.0
):
    """
    Predict the reflection of a barcode point on the flask surface.
    """

    dirs = mirror_dirs(angle_deg)
    s0 = hinge_offset_s0(angle_deg, gap_mm)

    P = surface_point(
        d_mm,
        phi_deg,
        radius_mm,
        flask_x_mm
    )

    nb = np.array([
        math.sin(math.radians(phi_deg)),
        math.cos(math.radians(phi_deg))
    ])

    result = {
        "direct": direct_view(camera, P, nb)
    }

    for side in ("left", "right"):
        result[side] = seen_in_mirror(
            camera,
            P,
            dirs[side],
            s0,
            width_mm,
            nb
        )

    return result


def axis_prediction(
    d_mm,
    camera,
    angle_deg=99.0,
    gap_mm=9.5,
    width_mm=100.0,
    flask_x_mm=0.0
):
    """Predict where the flask axis appears in each mirror."""

    dirs = mirror_dirs(angle_deg)
    s0 = hinge_offset_s0(angle_deg, gap_mm)

    P = np.array([flask_x_mm, d_mm])

    return {
        side: seen_in_mirror(
            camera,
            P,
            dirs[side],
            s0,
            width_mm
        )
        for side in ("left", "right")
    }


# ------------------------------------------------------------
# SOLVE BARCODE ANGLE
# ------------------------------------------------------------

def solve_phi(
    measured_d_edge,
    side,
    d_mm,
    camera,
    tol_mm=1.5,
    **kw
):
    """
    Find possible barcode surface angles matching a measured
    hinge-edge distance.
    """

    solutions = []

    for phi in np.arange(-180.0, 180.0, 0.1):

        result = predict(
            d_mm,
            phi,
            camera,
            **kw
        )[side]

        if result is None:
            continue

        if (
            result["on_glass"]
            and result["facing"]
            and abs(result["d_edge"] - measured_d_edge) <= tol_mm
        ):
            solutions.append(round(float(phi), 1))

    return solutions


# ------------------------------------------------------------
# SOLVE FLASK POSITION
# ------------------------------------------------------------

def solve_flask_position(
    left_mm,
    right_mm,
    camera,
    angle_deg=99.0,
    gap_mm=9.5,
    width_mm=100.0
):
    """
    Estimate flask-axis sideways position x and distance d
    using the two measured mirror readings.

    Two readings and two unknowns produce an exact fit by
    construction. A low residual does not independently
    validate the physical model.
    """

    C = np.asarray(camera, dtype=float)

    def cost(x, d):

        result = axis_prediction(
            d,
            C,
            angle_deg,
            gap_mm,
            width_mm,
            flask_x_mm=x
        )

        if result["left"] is None or result["right"] is None:
            return 1e12

        error_left = result["left"]["d_edge"] - left_mm
        error_right = result["right"]["d_edge"] - right_mm

        return error_left**2 + error_right**2

    best = 1e18
    best_x = 0.0
    best_d = 50.0

    # Coarse search.
    for x in np.arange(-60, 60.01, 1.0):
        for d in np.arange(0, 250.01, 1.0):

            value = cost(x, d)

            if value < best:
                best = value
                best_x = x
                best_d = d

    # Fine search.
    for step in (0.2, 0.05, 0.01):

        for _ in range(3):

            for x in np.arange(
                best_x - 10 * step,
                best_x + 10 * step + 1e-9,
                step
            ):

                for d in np.arange(
                    best_d - 10 * step,
                    best_d + 10 * step + 1e-9,
                    step
                ):

                    value = cost(x, d)

                    if value < best:
                        best = value
                        best_x = x
                        best_d = d

    return {
        "x_mm": float(best_x),
        "d_mm": float(best_d),
        "rms_residual_mm": float(math.sqrt(best / 2.0))
    }


# ------------------------------------------------------------
# SOLVE BARCODE MIDPOINT POSITION
# ------------------------------------------------------------

def solve_barcode_position(
    left_mm,
    right_mm,
    flask_d_mm,
    camera,
    angle_deg=99.0,
    gap_mm=9.5,
    width_mm=100.0,
    radius_mm=6.5,
    x_range=(-60.0, 60.0),
    phi_range=(-180.0, 180.0)
):
    """
    Estimate:
        x_mm: sideways flask-axis offset
        phi_deg: barcode midpoint angle around flask

    Inputs:
        left_mm: measured left-mirror barcode midpoint distance
        right_mm: measured right-mirror barcode midpoint distance
        flask_d_mm: known flask-axis distance along bisector
        camera: camera position [x, y]
        radius_mm: flask radius

    The solver uses two mirror readings to estimate two unknowns.
    A low residual alone does not validate the model.
    """

    C = np.asarray(camera, dtype=float)

    def evaluate(x_mm, phi_deg):

        result = predict(
            flask_d_mm,
            phi_deg,
            C,
            angle_deg=angle_deg,
            gap_mm=gap_mm,
            width_mm=width_mm,
            radius_mm=radius_mm,
            flask_x_mm=x_mm
        )

        L = result["left"]
        R = result["right"]

        if L is None or R is None:
            return None

        if not (L["on_glass"] and R["on_glass"]):
            return None

        if not (L["facing"] and R["facing"]):
            return None

        error_l = L["d_edge"] - left_mm
        error_r = R["d_edge"] - right_mm

        rms = math.sqrt(
            (error_l**2 + error_r**2) / 2.0
        )

        return {
            "x_mm": float(x_mm),
            "phi_deg": float(phi_deg),
            "predicted_left_mm": float(L["d_edge"]),
            "predicted_right_mm": float(R["d_edge"]),
            "left_error_mm": float(error_l),
            "right_error_mm": float(error_r),
            "rms_error_mm": float(rms)
        }

    best = None

    # Coarse search.
    for x in np.arange(
        x_range[0],
        x_range[1] + 0.5,
        0.5
    ):

        for phi in np.arange(
            phi_range[0],
            phi_range[1],
            1.0
        ):

            candidate = evaluate(x, phi)

            if candidate is not None:
                if (
                    best is None
                    or candidate["rms_error_mm"] < best["rms_error_mm"]
                ):
                    best = candidate

    if best is None:
        return None

    # Fine search around the best coarse result.
    best_x = best["x_mm"]
    best_phi = best["phi_deg"]

    for x_step, phi_step in (
        (0.1, 0.2),
        (0.02, 0.05)
    ):

        x_values = np.arange(
            best_x - 1.0,
            best_x + 1.0 + x_step / 2,
            x_step
        )

        phi_values = np.arange(
            best_phi - 2.0,
            best_phi + 2.0 + phi_step / 2,
            phi_step
        )

        for x in x_values:
            for phi in phi_values:

                candidate = evaluate(x, phi)

                if candidate is not None:
                    if (
                        candidate["rms_error_mm"]
                        < best["rms_error_mm"]
                    ):
                        best = candidate

        best_x = best["x_mm"]
        best_phi = best["phi_deg"]

    return best


# ------------------------------------------------------------
# GROUP ANGLE RESULTS
# ------------------------------------------------------------

def _group(vals):
    """Collapse sorted angles into contiguous ranges."""

    if not vals:
        return []

    out = []
    start = vals[0]
    prev = vals[0]

    for value in vals[1:]:

        if value - prev > 0.15:
            out.append((start, prev))
            start = value

        prev = value

    out.append((start, prev))

    return out


# ------------------------------------------------------------
# CALIBRATION-RECORD REVIEW (INDICATIVE ONLY)
# ------------------------------------------------------------

def load_calibration_record(path):
    """Load the two apparent barcode midpoint readings from a JSON record."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mirrors = data.get("mirrors", {})
    try:
        left = float(mirrors["left"]["barcode_distance_from_hinge_edge_mm"])
        right = float(mirrors["right"]["barcode_distance_from_hinge_edge_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: missing mirrors.left/right.barcode_distance_from_hinge_edge_mm"
        ) from exc

    return {
        "path": str(path),
        "left_mm": left,
        "right_mm": right,
        "mean_mm": (left + right) / 2.0,
        "angle_deg": data.get("geometry_stored_for_phase4_not_yet_used", {}).get(
            "included_angle_deg_nominal"
        ),
        "gap_top_mm": data.get("geometry_stored_for_phase4_not_yet_used", {}).get(
            "gap_top_mm_measured"
        ),
        "gap_bottom_mm": data.get("geometry_stored_for_phase4_not_yet_used", {}).get(
            "gap_bottom_mm_measured"
        ),
        "validated_with_ruler": data.get("validated_against_physical_ruler", False),
    }


def report_calibration_records(paths, positions_mm, camera, angle_deg, gap_mm, width_mm):
    """Compare axis predictions with JSON barcode-point readings, with caveats."""
    if len(paths) != len(positions_mm):
        raise ValueError("The number of --validation-positions-mm values must match --validate-json files.")

    print("\nCALIBRATION RECORD REVIEW (INDICATIVE, NOT A TRUE VALIDATION)")
    print("The JSON readings are apparent BARCODE-MIDPOINT positions; the model values below are FLASK-AXIS positions.")
    print("These are different geometric features, so their differences are diagnostic only.\n")
    print(f"{'position':>9} | {'record':>28} | {'obs L':>8} | {'pred L axis':>11} | {'obs R':>8} | {'pred R axis':>11} | {'mean Δ':>8}")
    print("-" * 112)

    rows = []
    for path, position in zip(paths, positions_mm):
        rec = load_calibration_record(path)
        pred = axis_prediction(
            float(position), camera, angle_deg, gap_mm, width_mm, flask_x_mm=0.0
        )
        left_pred = pred["left"]["d_edge"] if pred["left"] else float("nan")
        right_pred = pred["right"]["d_edge"] if pred["right"] else float("nan")
        mean_delta = rec["mean_mm"] - (left_pred + right_pred) / 2.0
        rows.append((rec, float(position), left_pred, right_pred, mean_delta))
        name = rec["path"].replace("\\", "/").split("/")[-1]
        print(
            f"{position:8.1f} | {name:>28.28} | {rec['left_mm']:8.1f} | "
            f"{left_pred:11.1f} | {rec['right_mm']:8.1f} | {right_pred:11.1f} | {mean_delta:+8.1f}"
        )

    print("\nPer-record left/right asymmetry (observed L minus R):")
    for rec, position, _, _, _ in rows:
        name = rec["path"].replace("\\", "/").split("/")[-1]
        print(f"  {position:5.1f} mm ({name}): {rec['left_mm'] - rec['right_mm']:+.1f} mm")

    print("\nRecord metadata says ruler-validated: " + ", ".join(
        "yes" if r[0]["validated_with_ruler"] else "no" for r in rows
    ))
    print("The JSON records note that 10 px/mm is rectification output resolution, not measurement accuracy.")
    print("Top/bottom gap wedge is reported in the JSON, but this top-down 2D model uses one mean gap; it does not model the vertical wedge.")
    print("Do not tune angle/camera/gap solely to force these six apparent barcode readings to match the axis predictions.\n")


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description="Top-down 2D reflection model for two mirrors."
    )

    # Camera position.
    ap.add_argument(
        "--camera-y-mm",
        type=float,
        required=True,
        help="Camera distance from the physical hinge reference along the bisector."
    )

    ap.add_argument(
        "--camera-x-mm",
        type=float,
        default=0.0,
        help="Camera sideways offset from bisector."
    )

    # Mirror geometry.
    ap.add_argument(
        "--angle-deg",
        type=float,
        default=99.0
    )

    ap.add_argument(
        "--gap-mm",
        type=float,
        default=9.5,
        help="Mean gap between inner glass edges; used to offset glass edges from the hinge-origin mirror intersection."
    )

    ap.add_argument(
        "--width-mm",
        type=float,
        default=100.0,
        help="Visible mirror glass width."
    )

    # Flask geometry.
    ap.add_argument(
        "--flask-od-mm",
        type=float,
        default=13.0
    )

    ap.add_argument(
        "--flask-mm",
        type=float,
        nargs="*",
        default=[30, 50, 70, 100],
        help="Flask-axis distances from the physical hinge reference to evaluate."
    )

    ap.add_argument(
        "--flask-x-mm",
        type=float,
        default=0.0,
        help="Sideways flask-axis offset."
    )

    # Barcode surface angles.
    ap.add_argument(
        "--phi",
        type=float,
        nargs="*",
        default=[],
        help="Barcode angles to evaluate."
    )

    # Solve flask position from two axis readings.
    ap.add_argument(
        "--solve-pos",
        type=float,
        nargs=2,
        metavar=("LEFT_MM", "RIGHT_MM"),
        default=None,
        help="Measured flask-axis distances in left and right mirrors."
    )

    # Solve barcode angle from one mirror.
    ap.add_argument(
        "--solve-left",
        type=float,
        default=None,
        help="Measured left-mirror hinge-edge distance."
    )

    ap.add_argument(
        "--solve-right",
        type=float,
        default=None,
        help="Measured right-mirror hinge-edge distance."
    )

    # Solve barcode midpoint position from both mirrors.
    ap.add_argument(
        "--solve-barcode",
        type=float,
        nargs=2,
        metavar=("LEFT_MM", "RIGHT_MM"),
        default=None,
        help="Measured barcode midpoint distances in left and right mirrors."
    )

    ap.add_argument(
        "--flask-distance-mm",
        type=float,
        default=None,
        help="Known flask-axis distance along mirror bisector for barcode solver."
    )

    ap.add_argument(
        "--validate-json",
        nargs="+",
        default=None,
        metavar="RECORD.json",
        help="Review calibration JSON files against model axis predictions (diagnostic only)."
    )

    ap.add_argument(
        "--validation-positions-mm",
        nargs="+",
        type=float,
        default=None,
        metavar="MM",
        help="Known hinge-referenced flask positions in the same order as --validate-json."
    )

    # IMPORTANT: Parse arguments BEFORE accessing a.solve_barcode,
    # a.camera_y_mm, or any other argument.
    a = ap.parse_args()

    # Camera position in millimetres.
    C = np.array([
        a.camera_x_mm,
        a.camera_y_mm
    ])

    # Shared geometry parameters.
    kw = dict(
        angle_deg=a.angle_deg,
        gap_mm=a.gap_mm,
        width_mm=a.width_mm,
        radius_mm=a.flask_od_mm / 2.0
    )

    # --------------------------------------------------------
    # Display model configuration.
    # --------------------------------------------------------

    print(
        f"Camera at (x={C[0]:.1f}, y={C[1]:.1f}) mm | "
        f"angle {a.angle_deg:.1f} deg | "
        f"gap {a.gap_mm:.1f} mm | "
        f"glass width {a.width_mm:.1f} mm | "
        f"s0 = {hinge_offset_s0(a.angle_deg, a.gap_mm):.1f} mm"
    )

    print("MODEL PREDICTIONS - not validated against measurements yet.\n")

    if a.validate_json is not None:
        if a.validation_positions_mm is None:
            raise SystemExit("ERROR: --validate-json requires --validation-positions-mm in the same file order.")
        report_calibration_records(
            a.validate_json, a.validation_positions_mm, C,
            a.angle_deg, a.gap_mm, a.width_mm
        )

    # --------------------------------------------------------
    # 1. Flask-axis prediction table.
    # --------------------------------------------------------

    print("Flask-axis image position measured from each mirror's inner glass edge")

    print(
        f"{'flask d (mm)':>13} | "
        f"{'left mirror':>14} | "
        f"{'right mirror':>14}"
    )

    for d in a.flask_mm:

        result = axis_prediction(
            d,
            C,
            a.angle_deg,
            a.gap_mm,
            a.width_mm,
            flask_x_mm=a.flask_x_mm
        )

        def format_axis(value):

            if value is None:
                return "no reflection"

            suffix = "" if value["on_glass"] else " (off glass)"

            return f"{value['d_edge']:7.1f}" + suffix

        print(
            f"{d:13.0f} | "
            f"{format_axis(result['left']):>14} | "
            f"{format_axis(result['right']):>14}"
        )

    # --------------------------------------------------------
    # 2. Solve flask position.
    # --------------------------------------------------------

    if a.solve_pos is not None:

        sol = solve_flask_position(
            a.solve_pos[0],
            a.solve_pos[1],
            C,
            a.angle_deg,
            a.gap_mm,
            a.width_mm
        )

        print("\nFLASK AXIS POSITION SOLUTION")

        print(
            f"Estimated sideways x = {sol['x_mm']:+.2f} mm"
        )

        print(
            f"Estimated distance d = {sol['d_mm']:.2f} mm"
        )

        print(
            f"Residual = {sol['rms_residual_mm']:.3f} mm"
        )

        print(
            "NOTE: Two readings / two unknowns produce an exact fit "
            "by construction. Compare the estimated distance with "
            "the physical flask position."
        )

        print(
            "A sideways camera offset and flask offset cannot be "
            "distinguished unless the camera position is known."
        )

    # --------------------------------------------------------
    # 3. Solve barcode midpoint position.
    # --------------------------------------------------------

    if a.solve_barcode is not None:

        if a.flask_distance_mm is None:
            raise SystemExit(
                "\nERROR: --solve-barcode requires --flask-distance-mm"
            )

        sol = solve_barcode_position(
            a.solve_barcode[0],
            a.solve_barcode[1],
            a.flask_distance_mm,
            C,
            angle_deg=a.angle_deg,
            gap_mm=a.gap_mm,
            width_mm=a.width_mm,
            radius_mm=a.flask_od_mm / 2.0
        )

        print("\nBARCODE MIDPOINT POSITION SOLUTION")

        if sol is None:

            print(
                "No valid solution found on both mirror surfaces."
            )

            print(
                "Check the measurements, camera position, "
                "flask distance, mirror angle and gap."
            )

        else:

            print(
                f"Known flask-axis distance: "
                f"{a.flask_distance_mm:.2f} mm"
            )

            print(
                f"Estimated sideways flask offset: "
                f"{sol['x_mm']:+.2f} mm"
            )

            print(
                f"Estimated barcode surface angle: "
                f"{sol['phi_deg']:+.2f} deg"
            )

            print(
                f"Predicted left mirror: "
                f"{sol['predicted_left_mm']:.2f} mm"
            )

            print(
                f"Measured left mirror: "
                f"{a.solve_barcode[0]:.2f} mm"
            )

            print(
                f"Predicted right mirror: "
                f"{sol['predicted_right_mm']:.2f} mm"
            )

            print(
                f"Measured right mirror: "
                f"{a.solve_barcode[1]:.2f} mm"
            )

            print(
                f"Left error: "
                f"{sol['left_error_mm']:+.2f} mm"
            )

            print(
                f"Right error: "
                f"{sol['right_error_mm']:+.2f} mm"
            )

            print(
                f"RMS error: "
                f"{sol['rms_error_mm']:.3f} mm"
            )

            print(
                "MODEL RESULT - requires validation against "
                "a physical flask position."
            )

    # --------------------------------------------------------
    # 4. Evaluate selected barcode angles.
    # --------------------------------------------------------

    for d in a.flask_mm:

        for phi in a.phi:

            result = predict(
                d,
                phi,
                C,
                **kw
            )

            def format_reflection(value):

                if value is None:
                    return "none"

                tag = ""

                if not value["on_glass"]:
                    tag += " off-glass"

                if not value["facing"]:
                    tag += " back-facing"

                return f"{value['d_edge']:.1f} mm{tag}"

            print(
                f"d={d:.0f} phi={phi:+.1f}: "
                f"direct={'yes' if result['direct'] else 'no'} | "
                f"left {format_reflection(result['left'])} | "
                f"right {format_reflection(result['right'])}"
            )

    # --------------------------------------------------------
    # 5. Solve barcode angle from individual mirror readings.
    # --------------------------------------------------------

    for side, value in (
        ("left", a.solve_left),
        ("right", a.solve_right)
    ):

        if value is None:
            continue

        for d in a.flask_mm:

            ranges = _group(
                solve_phi(
                    value,
                    side,
                    d,
                    C,
                    **kw
                )
            )

            txt = (
                ", ".join(
                    f"{lo:+.1f}..{hi:+.1f}"
                    for lo, hi in ranges
                )
                or "no match"
            )

            print(
                f"{side} mirror measured {value:.2f} mm, "
                f"flask d={d:.1f}: "
                f"candidate phi (deg) = {txt}"
            )


# ------------------------------------------------------------
# PROGRAM ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()