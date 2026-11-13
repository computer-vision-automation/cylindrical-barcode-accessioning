"""
Phase 6: 2D feature-matching experiment
--------------------------------------
Compares the known direct barcode crop from 10cm0 with the barcode area
visible in the 7cm+90 right-mirror rectification.

This is a separate diagnostic. It does NOT modify Phase 4 or Phase 5 files.

Run from PowerShell:
python "D:\tech\new\phase6_2d_match.py"

Outputs:
D:\tech\new\phase6_new\7cm+90_vs_10cm0\
    right_rectified_15pxmm.jpg
    mirror_barcode_roi.png
    mirror_barcode_roi_flipped.png
    orb_matches.png
    mirror_aligned_to_direct.png
    phase6_match_report.json

Notes:
- The barcode ROI is a manually selected starting region, not an automatic detector.
- 15 px/mm is a higher-resolution rewarp based on the local homography sampling
  density. It does not create new optical detail.
- ORB/RANSAC matching is diagnostic evidence only, not proof of barcode identity.
"""

import os
import json
import cv2
import numpy as np

ROOT = r"D:\tech\new"
IMAGE_PATH = os.path.join(ROOT, "samples", "7cm+90.jpg")
CALIB_PATH = os.path.join(ROOT, "phase(4)", "calibration_record7cm+90.json")
REFERENCE_PATH = os.path.join(
    ROOT, "phase5_new", "10cm0", "10cm0_direct_0.png"
)
OUT_DIR = os.path.join(ROOT, "phase6_new", "7cm+90_vs_10cm0")

# Coordinates measured on the 15 px/mm right-mirror rectified image.
# Format: x1, y1, x2, y2. Includes the barcode and nearby label features.
MIRROR_ROI_15PXMM = (900, 1500, 1090, 1830)
RECTIFIED_PX_PER_MM = 15.0

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image

def load_right_mirror_homography(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        calibration = json.load(f)
    H = np.asarray(
        calibration["mirrors"]["right"]["homography_photo_to_rectified"],
        dtype=np.float64,
    )
    base = calibration["rectified_size_px"]
    base_w = int(base["width"])
    base_h = int(base["height"])
    return H, base_w, base_h

def rewarp_right_mirror(photo, H_base, base_w, base_h, px_per_mm):
    # Calibration JSON stores the rectification at 10 px/mm.
    base_px_per_mm = 10.0
    scale = px_per_mm / base_px_per_mm
    S = np.diag([scale, scale, 1.0])
    H_scaled = S @ H_base
    out_w = int(round(base_w * scale))
    out_h = int(round(base_h * scale))
    rectified = cv2.warpPerspective(
        photo, H_scaled, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rectified, H_scaled

def orb_match(reference, mirror_roi):
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    mirror_gray = cv2.cvtColor(mirror_roi, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(
        nfeatures=1500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=5,
        patchSize=31,
        fastThreshold=5,
    )

    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_mirror, des_mirror = orb.detectAndCompute(mirror_gray, None)

    result = {
        "keypoints_reference": len(kp_ref),
        "keypoints_mirror": len(kp_mirror),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error_px": None,
        "homography_reference_to_mirror": None,
        "matches_image": None,
        "aligned_mirror": None,
    }

    if des_ref is None or des_mirror is None:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_ref, des_mirror, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    result["good_matches"] = len(good)

    if len(good) < 4:
        return result

    p_ref = np.float32(
        [kp_ref[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    p_mirror = np.float32(
        [kp_mirror[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        p_ref, p_mirror, cv2.RANSAC, 5.0
    )
    if H is None or mask is None:
        return result

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    projected = cv2.perspectiveTransform(p_ref, H).reshape(-1, 2)
    observed = p_mirror.reshape(-1, 2)
    errors = np.linalg.norm(projected - observed, axis=1)
    median_error = (
        float(np.median(errors[inlier_mask]))
        if inlier_count else None
    )

    result["inliers"] = inlier_count
    result["inlier_ratio"] = float(inlier_count / max(len(good), 1))
    result["median_reprojection_error_px"] = median_error
    result["homography_reference_to_mirror"] = H.tolist()

    draw_mask = mask.ravel().tolist()
    result["matches_image"] = cv2.drawMatches(
        reference, kp_ref,
        mirror_roi, kp_mirror,
        good, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=draw_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # H maps reference -> mirror ROI. Invert it to map mirror ROI -> reference.
    H_inv = np.linalg.inv(H)
    h, w = reference.shape[:2]
    result["aligned_mirror"] = cv2.warpPerspective(
        mirror_roi, H_inv, (w, h)
    )
    return result

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    photo = read_image(IMAGE_PATH)
    reference = read_image(REFERENCE_PATH)
    H_base, base_w, base_h = load_right_mirror_homography(CALIB_PATH)

    rectified, H_scaled = rewarp_right_mirror(
        photo, H_base, base_w, base_h, RECTIFIED_PX_PER_MM
    )

    x1, y1, x2, y2 = MIRROR_ROI_15PXMM
    if not (0 <= x1 < x2 <= rectified.shape[1]
            and 0 <= y1 < y2 <= rectified.shape[0]):
        raise ValueError(
            f"ROI {MIRROR_ROI_15PXMM} is outside rectified image "
            f"{rectified.shape[1]}x{rectified.shape[0]}"
        )

    mirror_roi = rectified[y1:y2, x1:x2].copy()
    mirror_flipped = cv2.flip(mirror_roi, 1)

    # Compare both possible mirror orientations; do not assume the flip.
    normal_result = orb_match(reference, mirror_roi)
    flipped_result = orb_match(reference, mirror_flipped)

    if flipped_result["inliers"] > normal_result["inliers"]:
        chosen_name = "horizontal_flip"
        chosen_roi = mirror_flipped
        chosen_result = flipped_result
    else:
        chosen_name = "unflipped"
        chosen_roi = mirror_roi
        chosen_result = normal_result

    cv2.imwrite(
        os.path.join(OUT_DIR, "right_rectified_15pxmm.jpg"),
        rectified,
    )
    cv2.imwrite(
        os.path.join(OUT_DIR, "mirror_barcode_roi.png"),
        mirror_roi,
    )
    cv2.imwrite(
        os.path.join(OUT_DIR, "mirror_barcode_roi_flipped.png"),
        mirror_flipped,
    )

    if chosen_result["matches_image"] is not None:
        cv2.imwrite(
            os.path.join(OUT_DIR, "orb_matches.png"),
            chosen_result["matches_image"],
        )
    if chosen_result["aligned_mirror"] is not None:
        cv2.imwrite(
            os.path.join(OUT_DIR, "mirror_aligned_to_direct.png"),
            chosen_result["aligned_mirror"],
        )

    report = {
        "phase": 6,
        "experiment": "2D ORB feature matching",
        "reference_image": REFERENCE_PATH,
        "mirror_source_image": IMAGE_PATH,
        "calibration_json": CALIB_PATH,
        "mirror_view": "right",
        "rectified_px_per_mm": RECTIFIED_PX_PER_MM,
        "mirror_roi_xyxy_on_scaled_rectification": list(MIRROR_ROI_15PXMM),
        "orientation_comparison": {
            "unflipped": {
                k: normal_result[k] for k in (
                    "keypoints_reference", "keypoints_mirror",
                    "good_matches", "inliers", "inlier_ratio",
                    "median_reprojection_error_px",
                )
            },
            "horizontal_flip": {
                k: flipped_result[k] for k in (
                    "keypoints_reference", "keypoints_mirror",
                    "good_matches", "inliers", "inlier_ratio",
                    "median_reprojection_error_px",
                )
            },
        },
        "chosen_orientation": chosen_name,
        "chosen_result": {
            k: chosen_result[k] for k in (
                "keypoints_reference", "keypoints_mirror",
                "good_matches", "inliers", "inlier_ratio",
                "median_reprojection_error_px",
                "homography_reference_to_mirror",
            )
        },
        "interpretation_warning": (
            "ORB/RANSAC feature agreement is a geometric matching diagnostic. "
            "It does not prove barcode identity, correct cylindrical unwrapping, "
            "or successful barcode decoding. The mirror ROI is manually selected."
        ),
    }

    with open(
        os.path.join(OUT_DIR, "phase6_match_report.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPHASE 6 - 2D MATCHING REPORT")
    print("=" * 55)
    print("Chosen orientation:", chosen_name)
    print("Good matches:", chosen_result["good_matches"])
    print("RANSAC inliers:", chosen_result["inliers"])
    print("Inlier ratio:", round(chosen_result["inlier_ratio"], 3))
    print("Median reprojection error:",
          chosen_result["median_reprojection_error_px"])
    print("Output folder:", OUT_DIR)
    print("\nThis is a feature-matching diagnostic, not a decode confirmation.")

if __name__ == "__main__":
    main()
