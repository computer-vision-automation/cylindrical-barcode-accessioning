
"""
run_pipeline.py - one command from photo to attempted decoded barcode.

Usage:
    python run_pipeline.py photo.jpg calibration.json --d-mm 70
    python run_pipeline.py photo.jpg calibration.json --d-mm 70 --out-dir output

Pipeline:
  1. Phase 5: detect barcode regions in direct view and both mirrors.
  2. Phase 6: estimate visible angular ranges and filter inconsistent views.
  3. Phase 7: map mirror crops onto a common angular axis.
     - X maps to angular position around the cylinder.
     - Y remains the original rectified-image Y coordinate.
     - Horizontal barcode bars are preserved.
     - Mirror overlaps use the sharper view.
     - Unseen angles remain blank.
  4. Decode the joined strip, individual strips, and raw crops.

Important:
  - Mirror reconstruction uses the original rectified vertical pixel scale.
  - Direct-view geometry is approximate, so the direct view is decoded separately
    rather than blended into the mirror reconstruction.
  - The 2D reflection model has an estimated angular accuracy of about +/-15-20 deg.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

import barcode_region_detector as brd
import phase6_angles as p6


HERE = os.path.dirname(os.path.abspath(__file__))

DEG_STEP = 0.5


def save_json(path, data):
    """Save JSON safely and consistently."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_phase5(photo, calib, out_dir):
    """Run barcode-region detection."""
    res = brd.run(
        photo,
        calib,
        out_dir,
        decode=False,
        min_score=0.5,
        top_per_view=0
    )

    stem = os.path.splitext(os.path.basename(photo))[0]
    det_json = os.path.join(out_dir, f"{stem}_detections.json")

    return res, det_json



def run_phase6(det_json, d_mm, extra=(), suffix="_angles.json"):
    """Run Phase 6 and load its separately named output."""

    angles_json = det_json.replace(
        "_detections.json",
        suffix
    )

    cmd = [
        sys.executable,
        os.path.join(HERE, "phase6_angles.py"),
        det_json,
        "--d-mm",
        str(d_mm),
        "--out",
        angles_json,
        *extra
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=HERE
    )

    if result.returncode != 0 or not os.path.exists(angles_json):
        raise RuntimeError(
            "Phase 6 failed.\n\n"
            + result.stdout
            + "\n"
            + result.stderr
        )

    with open(angles_json, encoding="utf-8") as f:
        angles_data = json.load(f)

    return angles_data, result.stdout


def get_crop(det, out_dir):
    """Load a detected crop."""
    crop_path = os.path.join(out_dir, det["crop_file"])
    crop = cv2.imread(crop_path)

    if crop is None:
        print(f"      WARNING: could not load crop: {crop_path}")

    return crop


def mirror_y_range(det, crop):
    """
    Return the crop's vertical range in original rectified-image coordinates.

    box_rectified_px provides the crop's original image origin.
    """
    x0, y0 = det["box_rectified_px"][:2]
    height = crop.shape[0]

    return int(y0), int(y0 + height)


def strip_for_view(
    entry,
    det,
    out_dir,
    cal,
    d_mm,
    cam,
    geo,
    angles,
    kw,
    y_bounds=None
):
    """
    Resample one view onto the common angular axis.

    Mirror views:
        X -> angular position
        Y -> original rectified-image Y coordinate

    This preserves horizontal barcode bars and their vertical sequence.

    Direct view:
        Uses its existing approximate geometry and native crop height.

    Returns:
        strip: image with shape (height, number_of_angles, 3)
        valid: boolean mask indicating usable angular columns
    """
    crop = get_crop(det, out_dir)

    if crop is None:
        return None, None

    ch, cw = crop.shape[:2]

    cols = np.full(len(angles), np.nan, dtype=np.float32)

    is_mirror = entry["view"] in ("left", "right")

    # ---------------------------------------------------------
    # MIRROR VIEW
    # ---------------------------------------------------------
    if is_mirror:

        phis, distances = p6.visible_curve(
            entry["view"],
            d_mm,
            cam,
            **kw
        )

        x0, y0 = det["box_rectified_px"][:2]

        # Convert each angular position into its corresponding
        # horizontal coordinate in the original rectified image.
        for k, angle in enumerate(angles):

            angle_wrapped = angle % 360.0

            if (
                angle_wrapped < phis.min()
                or angle_wrapped > phis.max()
            ):
                continue

            distance = float(
                np.interp(angle_wrapped, phis, distances)
            )

            if entry["view"] == "left":
                rectified_x = (
                    cal["width_mm"] - distance
                ) * cal["px_per_mm"]

            else:
                rectified_x = (
                    distance * cal["px_per_mm"]
                )

            cols[k] = rectified_x - x0

        # Use a shared original-image Y range for all mirror crops.
        if y_bounds is None:
            out_y0, out_y1 = mirror_y_range(det, crop)
        else:
            out_y0, out_y1 = y_bounds

        out_y0 = int(out_y0)
        out_y1 = int(out_y1)

        out_h = max(1, out_y1 - out_y0)

        # Convert global rectified Y coordinates into local crop
        # coordinates. Do not resize the mirror crop vertically.
        source_y = (
            np.arange(out_y0, out_y1, dtype=np.float32) - y0
        )

        valid_rows = (
            (source_y >= 0)
            & (source_y <= ch - 1)
        )

        map_y_column = np.clip(
            source_y,
            0,
            ch - 1
        )

    # ---------------------------------------------------------
    # DIRECT VIEW
    # ---------------------------------------------------------
    else:

        if geo is None:
            return None, None

        # Keep the original direct-view approximation.
        f_px = (
            geo["edge_height_px"]
            * (cam[1] - 4.1)
            / 150.0
        )

        radius_px = (
            kw["radius_mm"]
            * f_px
            / (cam[1] - d_mm)
        )

        x0 = det["box_photo_px"][0]

        for k, angle in enumerate(angles):

            wrapped_angle = (
                (angle + 180.0) % 360.0
            ) - 180.0

            if abs(wrapped_angle) < 89.0:

                cols[k] = (
                    geo["axis_x_px"]
                    + radius_px
                    * math.sin(math.radians(wrapped_angle))
                    - x0
                )

        # Direct-view vertical calibration is approximate.
        # Preserve the crop's native height rather than forcing
        # it into the mirror reconstruction's vertical scale.
        out_h = ch

        valid_rows = np.ones(
            out_h,
            dtype=bool
        )

        map_y_column = np.arange(
            out_h,
            dtype=np.float32
        )

    # ---------------------------------------------------------
    # COMMON RESAMPLING
    # ---------------------------------------------------------
    valid = (
        (~np.isnan(cols))
        & (cols >= 0)
        & (cols <= cw - 1)
    )

    if valid.sum() < 3 or not valid_rows.any():
        return None, None

    xs = np.where(
        valid,
        cols,
        0
    ).astype(np.float32)

    map_x = np.tile(
        xs[None, :],
        (out_h, 1)
    )

    map_y = np.tile(
        map_y_column[:, None],
        (1, len(angles))
    ).astype(np.float32)

    warped = cv2.remap(
        crop,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    strip = np.full(
        (out_h, len(angles), 3),
        128,
        dtype=np.uint8
    )

    strip[:, valid] = warped[:, valid]

    # Rows outside the original mirror crop must remain blank.
    strip[~valid_rows, :] = 128

    return strip, valid


def sharpness(strip, valid):
    """
    Calculate sharpness using only valid angular columns.

    The Laplacian is calculated before selecting columns, avoiding
    artificial edges caused by concatenating separated columns.
    """
    if strip is None or valid is None or not valid.any():
        return 0.0

    gray = cv2.cvtColor(
        strip,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(
        laplacian[:, valid].var()
    )


def decode_image(img):
    """Try four rotations using whichever decoders are installed."""
    hits = []

    if img is None or img.size == 0:
        return hits

    h, w = img.shape[:2]

    # Upscale small images for decoding.
    scale = max(
        1.0,
        900.0 / max(h, w)
    )

    if scale > 1.0:
        big = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
    else:
        big = img.copy()

    big = cv2.copyMakeBorder(
        big,
        40,
        40,
        40,
        40,
        cv2.BORDER_REPLICATE
    )

    rotations = (
        ("0", None),
        ("90", cv2.ROTATE_90_CLOCKWISE),
        ("180", cv2.ROTATE_180),
        ("270", cv2.ROTATE_90_COUNTERCLOCKWISE)
    )

    for rotation_name, rotation_code in rotations:

        if rotation_code is None:
            rotated = big
        else:
            rotated = cv2.rotate(
                big,
                rotation_code
            )

        try:
            results = brd.try_decode(rotated)
        except Exception as exc:
            print(
                f"      decoder warning at rotation "
                f"{rotation_name}: {exc}"
            )
            continue

        for result in results:
            result["rotation"] = rotation_name
            hits.append(result)

    return hits


def build_mirror_y_bounds(kept, detections, out_dir):
    """
    Calculate the union of the original Y ranges of all usable mirror crops.

    The resulting bounds allow left and right mirror crops to be aligned
    without vertically stretching each crop independently.
    """
    ranges = []

    for entry in kept:

        if entry["view"] not in ("left", "right"):
            continue

        det = detections[entry["detection_index"]]
        crop = get_crop(det, out_dir)

        if crop is None:
            continue

        ranges.append(
            mirror_y_range(det, crop)
        )

    if not ranges:
        return None

    return (
        min(r[0] for r in ranges),
        max(r[1] for r in ranges)
    )


def main():

    parser = argparse.ArgumentParser(
        description="Detect, reconstruct, and decode a cylindrical barcode."
    )

    parser.add_argument("photo")
    parser.add_argument("calib")

    parser.add_argument(
        "--d-mm",
        type=float,
        required=True,
        help="Flask-axis distance from the hinge reference in mm."
    )

    parser.add_argument(
        "--out-dir",
        default=None
    )

    parser.add_argument(
        "--camera-y-mm",
        type=float,
        default=240.0
    )

    args = parser.parse_args()

    if not os.path.isfile(args.photo):
        parser.error(f"Photo not found: {args.photo}")

    if not os.path.isfile(args.calib):
        parser.error(f"Calibration file not found: {args.calib}")

    stem = os.path.splitext(
        os.path.basename(args.photo)
    )[0]

    out_dir = args.out_dir or f"pipe_{stem}"

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    cal = brd.load_calibration(args.calib)

    cam = np.array([
        0.0,
        args.camera_y_mm
    ])

    # Keep the existing Phase 6 geometry parameters.
    kw = dict(
        angle_deg=99.0,
        gap_mm=9.5,
        width_mm=100.0,
        radius_mm=6.5
    )

    summary = {
        "photo": args.photo,
        "d_mm": args.d_mm,
        "views_used": [],
        "span_deg": None,
        "decoded": [],
        "attempts": []
    }

    # ---------------------------------------------------------
    # PHASE 5
    # ---------------------------------------------------------
    print("[1/4] detecting regions ...")

    res, det_json = run_phase5(
        args.photo,
        args.calib,
        out_dir
    )

    detections = res["detections"]

    print(
        f"      {len(detections)} candidate region(s)"
    )

    # ---------------------------------------------------------
    # PHASE 6
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # PHASE 6
    # ---------------------------------------------------------
    print("[2/4] estimating angles ...")

    ang, log = run_phase6(
        det_json,
        args.d_mm,
        extra=("--min-purity", "0.28", "--all"),
        suffix="_angles_test.json"
    )

    # TEMPORARY EXPERIMENT:
    # Reconstruct only detection #4 from the LEFT mirror.
    kept = [
        entry for entry in ang["angles"]
        if entry["view"] == "left"
        and entry["detection_index"] == 4
    ]

    if not kept:
        raise RuntimeError(
            "Detection left[4] was not found in Phase 6 test output."
        )

    selected = kept[0]

    start, end = selected["arc_deg"]
    width = selected["angular_width_deg"]

    # Use the selected candidate's own angular interval.
    ang["span_deg"] = [start, end, width]

    summary["span_deg"] = ang["span_deg"]

    for line in log.splitlines():
        if (
            line.startswith(("BARCODE", "SPAN", "["))
            and "NOTE" not in line
        ):
            print("      " + line)

    print(
        f"      TEST MODE: using only left[{selected['detection_index']}] "
        f"over {width:.1f} degrees"
    )

    if width <= 0:
        raise RuntimeError("Selected detection has an invalid angular width.")


    # ---------------------------------------------------------
    # COMMON ANGULAR AXIS
    # ---------------------------------------------------------
    start_angle, _, width = ang["span_deg"]

    angles = [
        start_angle + k * DEG_STEP
        for k in range(
            int(width / DEG_STEP) + 1
        )
    ]

    print(
        f"[3/4] un-mirroring and joining "
        f"{len(kept)} view(s) over {width} deg ..."
    )

    # Determine the shared original Y range for mirror crops.
    common_y_bounds = build_mirror_y_bounds(
        kept,
        detections,
        out_dir
    )

    if common_y_bounds is not None:

        print(
            f"      common mirror Y range: "
            f"{common_y_bounds[0]} to "
            f"{common_y_bounds[1]} px"
        )

    # ---------------------------------------------------------
    # BUILD INDIVIDUAL STRIPS
    # ---------------------------------------------------------
    all_strips = []

    for entry in kept:

        detection_index = entry["detection_index"]
        det = detections[detection_index]

        strip, valid = strip_for_view(
            entry,
            det,
            out_dir,
            cal,
            args.d_mm,
            cam,
            res.get("direct_geometry"),
            angles,
            kw,
            common_y_bounds
        )

        if strip is None:

            print(
                f"      {entry['view']}: "
                f"no usable pixels in the span"
            )

            continue

        score = sharpness(
            strip,
            valid
        )

        item = {
            "view": entry["view"],
            "strip": strip,
            "valid": valid,
            "sharpness": score,
            "detection": det,
            "arc_deg": entry.get("arc_deg")
        }

        all_strips.append(item)

        summary["views_used"].append({
            "view": entry["view"],
            "arc_deg": entry.get("arc_deg"),
            "sharpness": round(score, 1)
        })

        safe_view = str(
            entry["view"]
        ).replace(os.sep, "_")

        cv2.imwrite(
            os.path.join(
                out_dir,
                f"strip_{safe_view}_{detection_index}.png"
            ),
            strip
        )

        print(
            f"      {entry['view']}: "
            f"sharpness={score:.1f}, "
            f"valid columns={int(valid.sum())}"
        )

    if not all_strips:

        print("No view produced usable pixels.")

        save_json(
            os.path.join(
                out_dir,
                "pipeline_result.json"
            ),
            summary
        )

        return

    # ---------------------------------------------------------
    # JOIN MIRROR STRIPS
    # ---------------------------------------------------------
    mirror_strips = [
        item
        for item in all_strips
        if item["view"] in ("left", "right")
    ]

    direct_strips = [
        item
        for item in all_strips
        if item["view"] == "direct"
    ]

    if mirror_strips:

        # Highest sharpness gets priority in overlapping columns.
        mirror_strips.sort(
            key=lambda item: -item["sharpness"]
        )

        flat_height = mirror_strips[0]["strip"].shape[0]

        flat = np.full(
            (flat_height, len(angles), 3),
            128,
            dtype=np.uint8
        )

        mask = np.zeros(
            len(angles),
            dtype=np.uint8
        )

        for item in mirror_strips:

            strip = item["strip"]
            valid = item["valid"]

            if strip.shape[0] != flat_height:
                print(
                    f"      WARNING: skipping {item['view']} "
                    "because its strip height differs."
                )
                continue

            # Only fill angular columns not already supplied
            # by a sharper mirror view.
            take = valid & (mask == 0)

            flat[:, take] = strip[:, take]
            mask[take] = 255

        cv2.imwrite(
            os.path.join(
                out_dir,
                "flat_strip.png"
            ),
            flat
        )

        cv2.imwrite(
            os.path.join(
                out_dir,
                "flat_mask.png"
            ),
            np.tile(
                mask,
                (30, 1)
            )
        )

        filled_degrees = (
            int((mask > 0).sum()) * DEG_STEP
        )

        print(
            f"      saved flat_strip.png "
            f"({filled_degrees:.1f} of {width:.1f} deg filled)"
        )

    else:

        # If no mirror strip survived, use the best direct strip
        # as a fallback. This is not a mirror-based reconstruction.
        direct_strips.sort(
            key=lambda item: -item["sharpness"]
        )

        best_direct = direct_strips[0]

        flat = best_direct["strip"].copy()

        mask = (
            best_direct["valid"].astype(np.uint8) * 255
        )

        cv2.imwrite(
            os.path.join(
                out_dir,
                "flat_strip.png"
            ),
            flat
        )

        cv2.imwrite(
            os.path.join(
                out_dir,
                "flat_mask.png"
            ),
            np.tile(
                mask,
                (30, 1)
            )
        )

        print(
            "      no mirror strips available; "
            "saved best direct-view strip as fallback."
        )

    # ---------------------------------------------------------
    # PHASE 4: DECODE
    # ---------------------------------------------------------
    print("[4/4] decoding ...")

    tried = [
        ("joined strip", flat)
    ]

    # Try every individual view independently.
    for item in all_strips:

        view = item["view"]
        strip = item["strip"]
        det = item["detection"]

        tried.append(
            (f"{view} strip", strip)
        )

        crop = get_crop(
            det,
            out_dir
        )

        if crop is not None:

            tried.append(
                (f"{view} raw crop", crop)
            )

            tried.append(
                (
                    f"{view} raw crop (flipped)",
                    cv2.flip(crop, 1)
                )
            )

    # Decode every candidate image.
    for name, image in tried:

        hits = decode_image(image)

        summary["attempts"].append({
            "image": name,
            "hits": len(hits)
        })

        for hit in hits:

            summary["decoded"].append({
                "from": name,
                **hit
            })

        if hits:

            print(
                f"      {name}: DECODED "
                f"{hits[0].get('text')} "
                f"({hits[0].get('decoder')}, "
                f"rotation {hits[0]['rotation']})"
            )

        else:

            print(
                f"      {name}: no decode"
            )

    # ---------------------------------------------------------
    # SAVE FINAL REPORT
    # ---------------------------------------------------------
    result_path = os.path.join(
        out_dir,
        "pipeline_result.json"
    )

    save_json(
        result_path,
        summary
    )

    if not summary["decoded"]:

        print(
            "      no decoder read a barcode. "
            "For the strongest decoding option, install "
            "zxing-cpp: pip install zxing-cpp"
        )

    print()
    print("Pipeline complete.")
    print(f"Results saved in: {out_dir}/")
    print(f"Report: {result_path}")


if __name__ == "__main__":
    main()
