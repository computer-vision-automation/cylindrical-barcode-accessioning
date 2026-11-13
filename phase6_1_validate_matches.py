
import cv2
import numpy as np
import json
from pathlib import Path

# ============================================================
# PHASE 6.1: 2D BARCODE MATCH VALIDATION
# ============================================================

ROOT = Path(r"D:\tech\new")

REFERENCE_PATH = (
    ROOT / "phase5_new" / "10cm0" / "10cm0_direct_0.png"
)

MIRROR_PATH = (
    ROOT / "phase6_new" / "7cm+90_vs_10cm0"
    / "mirror_barcode_roi.png"
)

OUT_DIR = ROOT / "phase6_new" / "validation_6_1"

OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_BARCODE = "8901425076811"

# ORB settings
ORB_FEATURES = 2500
RATIO_TEST = 0.75
RANSAC_THRESHOLD = 4.0

# Approximate barcode-focused regions.
# Coordinates: x, y, width, height.
# Adjust these if your crop dimensions differ.
REFERENCE_ROI = (25, 5, 160, 340)
MIRROR_ROI = (60, 55, 115, 225)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load image: {path}"
        )

    return image


def crop_roi(image, roi):

    x, y, w, h = roi

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"ROI {roi} is outside image dimensions "
            f"{image.shape[1]}x{image.shape[0]}"
        )

    return image[y1:y2, x1:x2].copy()


# ============================================================
# FEATURE EXTRACTION AND MATCHING
# ============================================================

def create_orb():

    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=5,
        patchSize=31,
        fastThreshold=5
    )


def calculate_spatial_coverage(points, shape):

    h, w = shape[:2]

    if len(points) == 0:
        return {
            "count": 0,
            "occupied_grid_cells": 0,
            "convex_hull_fraction": 0
        }

    points = np.asarray(points, dtype=np.float32)

    hull = cv2.convexHull(
        points.reshape(-1, 1, 2)
    )

    hull_area = cv2.contourArea(hull)

    image_area = w * h

    # Divide image into 4 x 4 cells.
    grid = set()

    for x, y in points:

        gx = min(3, max(0, int(x / w * 4)))
        gy = min(3, max(0, int(y / h * 4)))

        grid.add((gx, gy))

    return {
        "count": len(points),
        "occupied_grid_cells": len(grid),
        "convex_hull_fraction": (
            float(hull_area / image_area)
            if image_area else 0
        )
    }


def match_images(reference, mirror):

    orb = create_orb()

    ref_gray = cv2.cvtColor(
        reference, cv2.COLOR_BGR2GRAY
    )

    mir_gray = cv2.cvtColor(
        mirror, cv2.COLOR_BGR2GRAY
    )

    kp1, des1 = orb.detectAndCompute(
        ref_gray, None
    )

    kp2, des2 = orb.detectAndCompute(
        mir_gray, None
    )

    result = {
        "keypoints_reference": len(kp1),
        "keypoints_mirror": len(kp2),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0,
        "median_error": None,
        "homography": None,
        "reference_coverage": None,
        "mirror_coverage": None,
        "visualization": None,
        "warped_mirror": None,
        "overlay": None
    }

    if des1 is None or des2 is None:

        print("Not enough ORB descriptors.")

        return result

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pairs = matcher.knnMatch(
        des1, des2, k=2
    )

    good = []

    for pair in pairs:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    result["good_matches"] = len(good)

    if len(good) < 4:

        print("Not enough good matches.")

        return result

    src = np.float32([
        kp1[m.queryIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp2[m.trainIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:

        print("Homography estimation failed.")

        return result

    inliers = mask.ravel().astype(bool)

    src_points = src.reshape(-1, 2)
    dst_points = dst.reshape(-1, 2)

    projected = cv2.perspectiveTransform(
        src, H
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - dst_points,
        axis=1
    )

    inlier_errors = errors[inliers]

    result["inliers"] = int(inliers.sum())

    result["inlier_ratio"] = float(
        inliers.sum() / len(good)
    )

    if len(inlier_errors):

        result["median_error"] = float(
            np.median(inlier_errors)
        )

    result["homography"] = H.tolist()

    result["reference_coverage"] = (
        calculate_spatial_coverage(
            src_points[inliers],
            reference.shape
        )
    )

    result["mirror_coverage"] = (
        calculate_spatial_coverage(
            dst_points[inliers],
            mirror.shape
        )
    )

    # Draw only RANSAC inliers.
    result["visualization"] = cv2.drawMatches(
        reference,
        kp1,
        mirror,
        kp2,
        good,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=mask.ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # H maps reference -> mirror.
    # Invert H to map mirror -> reference.
    H_inv = np.linalg.inv(H)

    h, w = reference.shape[:2]

    warped = cv2.warpPerspective(
        mirror,
        H_inv,
        (w, h),
        flags=cv2.INTER_CUBIC
    )

    result["warped_mirror"] = warped

    result["overlay"] = cv2.addWeighted(
        reference,
        0.5,
        warped,
        0.5,
        0
    )

    return result


# ============================================================
# OPTIONAL BARCODE DECODING
# ============================================================

def try_zxing(image):

    try:
        import zxingcpp

    except ImportError:

        return {
            "available": False,
            "message": "zxingcpp is not installed.",
            "decoded": []
        }

    variants = {
        "original": image,
        "horizontal_flip": cv2.flip(image, 1),
        "clockwise": cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        ),
        "counterclockwise": cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )
    }

    decoded_results = []

    for name, variant in variants.items():

        try:

            results = zxingcpp.read_barcodes(
                variant
            )

            for item in results:

                decoded_results.append({
                    "variant": name,
                    "format": str(item.format),
                    "value": str(item.text)
                })

        except Exception as error:

            print("ZXing error:", error)

    return {
        "available": True,
        "expected_value": EXPECTED_BARCODE,
        "decoded": decoded_results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPHASE 6.1 - BARCODE MATCH VALIDATION")
    print("=" * 55)

    reference = load_image(
        REFERENCE_PATH
    )

    mirror = load_image(
        MIRROR_PATH
    )

    print("Reference dimensions:", reference.shape)
    print("Mirror dimensions:", mirror.shape)

    # Mirror reflection should be tested in reversed orientation.
    mirror_flipped = cv2.flip(
        mirror, 1
    )

    # --------------------------------------------------------
    # TEST A: COMPLETE CROPS
    # --------------------------------------------------------

    print("\nTEST A: FULL IMAGE MATCHING")

    full_result = match_images(
        reference,
        mirror_flipped
    )

    # --------------------------------------------------------
    # TEST B: BARCODE-FOCUSED CROPS
    # --------------------------------------------------------

    print("\nTEST B: BARCODE-FOCUSED MATCHING")

    reference_barcode = crop_roi(
        reference,
        REFERENCE_ROI
    )

    mirror_barcode = crop_roi(
        mirror_flipped,
        MIRROR_ROI
    )

    barcode_result = match_images(
        reference_barcode,
        mirror_barcode
    )

    # --------------------------------------------------------
    # SAVE IMAGES
    # --------------------------------------------------------

    def save_result(name, result):

        if result["visualization"] is not None:

            cv2.imwrite(
                str(OUT_DIR / f"{name}_matches.png"),
                result["visualization"]
            )

        if result["warped_mirror"] is not None:

            cv2.imwrite(
                str(OUT_DIR / f"{name}_warped.png"),
                result["warped_mirror"]
            )

        if result["overlay"] is not None:

            cv2.imwrite(
                str(OUT_DIR / f"{name}_overlay.png"),
                result["overlay"]
            )

    save_result("full_crop", full_result)

    save_result("barcode_only", barcode_result)

    cv2.imwrite(
        str(OUT_DIR / "reference_barcode_crop.png"),
        reference_barcode
    )

    cv2.imwrite(
        str(OUT_DIR / "mirror_barcode_crop.png"),
        mirror_barcode
    )

    # --------------------------------------------------------
    # OPTIONAL DECODE
    # --------------------------------------------------------

    if barcode_result["warped_mirror"] is not None:

        decode_result = try_zxing(
            barcode_result["warped_mirror"]
        )

    else:

        decode_result = {
            "available": False,
            "message": "No valid homography; decoding skipped.",
            "decoded": []
        }

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    def clean_result(result):

        return {
            key: value
            for key, value in result.items()
            if key not in (
                "visualization",
                "warped_mirror",
                "overlay"
            )
        }

    report = {
        "phase": "6.1",
        "reference": str(REFERENCE_PATH),
        "mirror_roi": str(MIRROR_PATH),
        "mirror_orientation": "horizontal_flip",

        "reference_roi": REFERENCE_ROI,
        "mirror_roi_coordinates": MIRROR_ROI,

        "full_image_result": clean_result(
            full_result
        ),

        "barcode_focused_result": clean_result(
            barcode_result
        ),

        "decode_result": decode_result,

        "interpretation_warning": (
            "ORB matching and homography do not prove "
            "correct cylindrical reconstruction or barcode identity."
        )
    }

    report_path = (
        OUT_DIR / "phase6_1_validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    for name, result in [
        ("FULL IMAGE", full_result),
        ("BARCODE FOCUSED", barcode_result)
    ]:

        print("\n", name)

        print(
            "Good matches:",
            result["good_matches"]
        )

        print(
            "RANSAC inliers:",
            result["inliers"]
        )

        print(
            "Inlier ratio:",
            round(result["inlier_ratio"], 3)
        )

        print(
            "Median error:",
            result["median_error"]
        )

        print(
            "Spatial coverage:",
            result["mirror_coverage"]
        )

    print("\nZXING RESULTS")

    print(json.dumps(
        decode_result,
        indent=2
    ))

    print("\nAll results saved to:")
    print(OUT_DIR)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
