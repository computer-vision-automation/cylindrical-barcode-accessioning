
"""
barcode_region_detector_v2.py
Phase 5 V2: Barcode localisation with duplicate-fragment suppression.

Keep barcode_region_detector(1).py in the same folder.
The original detector handles calibration, rectification, detection,
cropping and decoding. This V2 wrapper cleans up its detections.
"""

import argparse
import importlib.util
import json
import os
import cv2


BASE_FILE = "barcode_region_detector(1).py"


def load_base_detector():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BASE_FILE)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cannot find {BASE_FILE}. Keep it in the same folder "
            "as barcode_region_detector_v2.py."
        )

    spec = importlib.util.spec_from_file_location("base_barcode_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def intersection_over_smaller(a, b):
    """Intersection area / area of the smaller bounding box."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))

    intersection = ix * iy
    smaller_area = min(aw * ah, bw * bh)

    return intersection / float(smaller_area + 1e-9)


def suppress_contained_duplicates(detections, threshold=0.90):
    """
    Suppress smaller detections almost entirely contained in a larger,
    stronger detection from the same mirror.
    """
    result = []

    for side in ("left", "right"):
        group = [d for d in detections if d.get("view") == side]

        # Process larger regions first, preferring stronger scores on ties.
        group.sort(
            key=lambda d: (
                -(
                    d["box_rectified_px"][2]
                    * d["box_rectified_px"][3]
                ),
                -d["score"],
            )
        )

        accepted = []

        for candidate in group:
            box = candidate.get("box_rectified_px")

            if not box:
                accepted.append(candidate)
                continue

            duplicate = False

            for parent in accepted:
                parent_box = parent.get("box_rectified_px")

                if not parent_box:
                    continue

                overlap = intersection_over_smaller(box, parent_box)

                # The parent must be at least as strong, allowing a small
                # score tolerance for detector variation.
                strong_enough = (
                    parent.get("score", 0)
                    >= candidate.get("score", 0) - 0.05
                )

                if overlap >= threshold and strong_enough:
                    duplicate = True
                    break

            if not duplicate:
                accepted.append(candidate)

        result.extend(accepted)

    # Preserve direct-view detections; this cleanup targets mirror views.
    result.extend(
        d for d in detections
        if d.get("view") not in ("left", "right")
    )

    return result


def filter_small_regions(detections, min_height_mm=12.0):
    """
    Remove tiny mirror proposals using their rectified physical height.

    Set min_height_mm=0 to disable this filter.
    """
    if min_height_mm <= 0:
        return detections

    kept = []

    for d in detections:
        if d.get("view") not in ("left", "right"):
            kept.append(d)
            continue

        mm_box = d.get("box_rectified_mm")

        if not mm_box:
            kept.append(d)
            continue

        height_mm = mm_box[3]

        if height_mm >= min_height_mm:
            kept.append(d)

    return kept


def apply_top_per_view(detections, limit):
    """Keep at most N highest-scoring regions per view. Zero means unlimited."""
    detections = sorted(
        detections,
        key=lambda d: -d.get("score", 0)
    )

    if limit <= 0:
        return detections

    counts = {}
    kept = []

    for d in detections:
        view = d.get("view", "unknown")
        counts[view] = counts.get(view, 0) + 1

        if counts[view] <= limit:
            kept.append(d)

    return kept


def run(args):
    base = load_base_detector()

    # Pass command-line settings to the original detector's global filters.
    base.MIN_PURITY = args.min_purity
    base.MAX_HINGE_MM = args.max_hinge_mm
    base.EXTENT_MIN_CONTRAST = args.extent_contrast
    base.DEBUG = args.debug

    direct_roi = tuple(args.direct_roi) if args.direct_roi else None

    # Ask the original detector to return every score-qualified candidate.
    # V2 performs its own duplicate and size filtering afterwards.
    result = base.run(
        image_path=args.image,
        calib_path=args.calib,
        out_dir=args.out_dir,
        direct_roi=direct_roi,
        decode=not args.no_decode,
        min_score=args.min_score,
        ignore_below_mm=args.ignore_below_mm or None,
        top_per_view=0,
    )

    original_count = len(result["detections"])

    # V2 cleanup.
    detections = suppress_contained_duplicates(
        result["detections"],
        threshold=args.containment,
    )

    detections = filter_small_regions(
        detections,
        min_height_mm=args.min_region_height_mm,
    )

    detections = apply_top_per_view(
        detections,
        args.top_per_view,
    )

    result["detections"] = detections
    result["v2_cleanup"] = {
        "original_candidate_count": original_count,
        "final_candidate_count": len(detections),
        "containment_threshold": args.containment,
        "minimum_region_height_mm": args.min_region_height_mm,
        "note": (
            "Duplicate suppression and minimum-height filtering are "
            "heuristics. Surviving regions are not guaranteed to be barcodes."
        ),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    json_path = os.path.join(
        args.out_dir,
        f"{stem}_detections.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Regenerate the annotated image using only surviving detections.
    image = cv2.imread(args.image)

    if image is not None:
        calibration = (
    base.load_calibration(args.calib, image.shape)
    if args.calib else None
)

        roi = result.get("direct_roi_px")

        annotated = base.annotate(
            image,
            calibration,
            detections,
            tuple(roi) if roi else None,
        )

        cv2.imwrite(
            os.path.join(args.out_dir, f"{stem}_annotated.jpg"),
            annotated,
        )

    print("\nPHASE 5 V2 DETECTION REPORT")
    print("=" * 55)
    print(f"Original candidates : {original_count}")
    print(f"After V2 cleanup   : {len(detections)}")

    for i, d in enumerate(detections):
        extra = ""

        if "hinge_edge_distance_mm" in d:
            h = d["hinge_edge_distance_mm"]
            extra = (
                f" hinge={h['near']:.2f}-{h['far']:.2f} mm"
                f" centre={h['centre']:.2f} mm"
            )

        print(
            f"[{i}] {d['view']:6s}"
            f" score={d['score']:.2f}"
            f" orientation={d.get('orientation', '-')}"
            f"{extra}"
        )

    print("\nSaved updated JSON and annotation to:", args.out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 V2 barcode detector"
    )

    parser.add_argument("image")
    parser.add_argument("--calib")
    parser.add_argument("--out-dir", default="phase5_v2")
    parser.add_argument(
        "--direct-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
    )

    parser.add_argument("--ignore-below-mm", type=float, default=124.0)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--min-purity", type=float, default=0.0)
    parser.add_argument("--max-hinge-mm", type=float, default=93.5)
    parser.add_argument("--extent-contrast", type=float, default=0.33)

    parser.add_argument(
        "--min-region-height-mm",
        type=float,
        default=12.0,
        help="Minimum mirror candidate height. Set 0 to disable.",
    )

    parser.add_argument(
        "--containment",
        type=float,
        default=0.90,
        help="Intersection-over-smaller threshold for duplicate suppression.",
    )

    parser.add_argument(
        "--top-per-view",
        type=int,
        default=0,
        help="Maximum detections per view. 0 keeps all.",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-decode", action="store_true")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
