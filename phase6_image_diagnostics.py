
from pathlib import Path
import cv2
import numpy as np
import csv

# ============================================================
# PHASE 6: IMAGE-BASED BARCODE ALIGNMENT DIAGNOSTICS
# ============================================================

OUT = Path("phase6_image_diagnostics")
OUT.mkdir(exist_ok=True)

crops = {
    "right_main": Path("7cm+90_right_5.png"),
    "left": Path("phase5_test/7cm+90_left_4.png"),
    "right_secondary_6": Path("phase5_test/7cm+90_right_6.png"),
    "right_secondary_7": Path("phase5_test/7cm+90_right_7.png"),
}

images = {}
profiles = {}
report = []
comparisons = []


# ============================================================
# 1. PROFILE GENERATION
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-8)


def make_profile(gray):
    """
    Generate a vertical barcode-pattern profile.

    Barcode bars are horizontal, so dark-pixel density
    is measured row by row along the vertical sequence.
    Original vertical resolution is preserved.
    """

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    dark_fraction = (binary > 0).mean(axis=1)

    return np.asarray(dark_fraction, dtype=np.float64)


# ============================================================
# 2. SCALE-AWARE PROFILE COMPARISON
# ============================================================

def best_correlation(a, b):

    best = {
        "score": -2.0,
        "lag": 0,
        "orientation": "normal",
        "scale": 1.0,
        "overlap": 0,
        "overlap_pct": 0.0,
        "candidate_coverage_pct": 0.0,
    }

    scale_factors = np.arange(0.5, 6.01, 0.1)

    for orientation, original in (
        ("normal", b),
        ("vertical_reverse", b[::-1])
    ):

        for scale in scale_factors:

            new_length = max(
                1,
                int(round(len(original) * scale))
            )

            candidate = cv2.resize(
                original.reshape(-1, 1),
                (1, new_length),
                interpolation=cv2.INTER_LINEAR
            ).ravel()

            min_overlap = 64

            if min(len(a), len(candidate)) < min_overlap:
                continue

            for lag in range(
                -len(candidate) + min_overlap,
                len(a) - min_overlap + 1
            ):

                a_start = max(0, lag)
                b_start = max(0, -lag)

                overlap = min(
                    len(a) - a_start,
                    len(candidate) - b_start
                )

                if overlap < min_overlap:
                    continue

                aa = a[a_start:a_start + overlap]
                bb = candidate[b_start:b_start + overlap]

                if aa.std() < 1e-8 or bb.std() < 1e-8:
                    continue

                score = float(np.corrcoef(aa, bb)[0, 1])

                if np.isfinite(score) and score > best["score"]:

                    best = {
                        "score": score,
                        "lag": lag,
                        "orientation": orientation,
                        "scale": float(scale),
                        "overlap": overlap,

                        # Percentage of the main reference covered.
                        "overlap_pct": round(
                            100 * overlap / len(a), 1
                        ),

                        # Percentage of the resized candidate covered.
                        "candidate_coverage_pct": round(
                            100 * overlap / len(candidate), 1
                        ),
                    }

    return best


# ============================================================
# 3. ALIGNMENT SECTION EXTRACTION
# ============================================================

def extract_aligned_sections(
    reference_gray,
    candidate_gray,
    result
):
    """
    Extract the sections corresponding to the best
    vertical profile alignment.

    This is a diagnostic alignment, not a calibrated
    physical mirror transformation.
    """

    candidate = candidate_gray.copy()

    if result["orientation"] == "vertical_reverse":
        candidate = candidate[::-1, :]

    new_height = max(
        1,
        int(round(candidate.shape[0] * result["scale"]))
    )

    candidate = cv2.resize(
        candidate,
        (candidate.shape[1], new_height),
        interpolation=cv2.INTER_LINEAR
    )

    lag = result["lag"]

    a_start = max(0, lag)
    b_start = max(0, -lag)

    overlap = min(
        reference_gray.shape[0] - a_start,
        candidate.shape[0] - b_start
    )

    if overlap <= 0:
        return None, None

    ref_section = reference_gray[
        a_start:a_start + overlap
    ]

    candidate_section = candidate[
        b_start:b_start + overlap
    ]

    return ref_section, candidate_section


# ============================================================
# 4. SAVE ENLARGED ALIGNMENT PREVIEW
# ============================================================

def save_alignment_preview(
    reference_gray,
    candidate_gray,
    result,
    name
):

    ref_section, candidate_section = extract_aligned_sections(
        reference_gray,
        candidate_gray,
        result
    )

    if ref_section is None:
        return

    target_height = 500

    ref_width = max(
        1,
        int(
            ref_section.shape[1]
            * target_height
            / ref_section.shape[0]
        )
    )

    candidate_width = max(
        1,
        int(
            candidate_section.shape[1]
            * target_height
            / candidate_section.shape[0]
        )
    )

    ref_preview = cv2.resize(
        ref_section,
        (ref_width, target_height),
        interpolation=cv2.INTER_NEAREST
    )

    candidate_preview = cv2.resize(
        candidate_section,
        (candidate_width, target_height),
        interpolation=cv2.INTER_NEAREST
    )

    gap = 20
    header = 35

    canvas = np.full(
        (
            target_height + header,
            ref_width + candidate_width + gap
        ),
        255,
        dtype=np.uint8
    )

    x2 = ref_width + gap

    canvas[
        header:,
        :ref_width
    ] = ref_preview

    canvas[
        header:,
        x2:x2 + candidate_width
    ] = candidate_preview

    cv2.putText(
        canvas,
        "RIGHT MAIN MATCHED SECTION",
        (5, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        0,
        1
    )

    cv2.putText(
        canvas,
        name,
        (x2, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        0,
        1
    )

    cv2.imwrite(
        str(OUT / f"alignment_{name}.png"),
        canvas
    )


# ============================================================
# 5. SAVE 2D SIDE-BY-SIDE AND OVERLAY PREVIEWS
# ============================================================

def save_2d_overlay(
    reference_gray,
    candidate_gray,
    result,
    name
):

    ref_section, candidate_section = extract_aligned_sections(
        reference_gray,
        candidate_gray,
        result
    )

    if ref_section is None:
        return

    # Save the two matched sections side by side.
    height = ref_section.shape[0]

    # Resize candidate width for visual comparison only.
    # This is NOT a physical geometric calibration.
    candidate_for_overlay = cv2.resize(
        candidate_section,
        (ref_section.shape[1], height),
        interpolation=cv2.INTER_LINEAR
    )

    side_by_side = np.hstack((
        ref_section,
        candidate_for_overlay
    ))

    overlay = cv2.addWeighted(
        ref_section,
        0.5,
        candidate_for_overlay,
        0.5,
        0
    )

    cv2.imwrite(
        str(OUT / f"alignment_2d_{name}.png"),
        side_by_side
    )

    cv2.imwrite(
        str(OUT / f"overlay_2d_{name}.png"),
        overlay
    )


# ============================================================
# 6. LOAD AND INSPECT IMAGES
# ============================================================

for name, path in crops.items():

    if not path.is_file():
        print("MISSING:", path)
        continue

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("UNREADABLE:", path)
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    images[name] = (image, gray)
    profiles[name] = make_profile(gray)

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    report.append({
        "crop": name,
        "path": str(path.resolve()),
        "width": gray.shape[1],
        "height": gray.shape[0],
        "mean": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "laplacian_variance": round(float(lap), 2),
    })


# ============================================================
# 7. PRINT IMAGE QUALITY REPORT
# ============================================================

print("\nPHASE 6 IMAGE-BASED DIAGNOSTICS")
print("=" * 65)

for item in report:

    print(
        f"{item['crop']}: "
        f"{item['width']}x{item['height']} | "
        f"contrast={item['contrast']} | "
        f"sharpness_indicator={item['laplacian_variance']}"
    )


# ============================================================
# 8. COMPARE MAIN RIGHT CROP WITH OTHER VIEWS
# ============================================================

print("\nVERTICAL BARCODE-PATTERN COMPARISON")
print("Correlation is diagnostic only, not proof of matching content.")
print("-" * 65)

if "right_main" in profiles:

    reference_gray = images["right_main"][1]

    for name in profiles:

        if name == "right_main":
            continue

        result = best_correlation(
            profiles["right_main"],
            profiles[name]
        )

        comparisons.append({
            "reference": "right_main",
            "candidate": name,
            "correlation": round(result["score"], 4),
            "lag": result["lag"],
            "orientation": result["orientation"],
            "vertical_scale": round(result["scale"], 2),
            "overlap_samples": result["overlap"],
            "main_overlap_percent": result["overlap_pct"],
            "candidate_coverage_percent":
                result["candidate_coverage_pct"],
        })

        print(
            f"right_main vs {name}: "
            f"correlation={result['score']:.4f}, "
            f"lag={result['lag']}, "
            f"orientation={result['orientation']}, "
            f"scale={result['scale']:.2f}, "
            f"main_overlap={result['overlap_pct']}%, "
            f"candidate_coverage="
            f"{result['candidate_coverage_pct']}%"
        )

        # Save visual alignment previews.
        candidate_gray = images[name][1]

        save_alignment_preview(
            reference_gray,
            candidate_gray,
            result,
            name
        )

        save_2d_overlay(
            reference_gray,
            candidate_gray,
            result,
            name
        )


# ============================================================
# 9. SAVE CSV REPORTS
# ============================================================

if report:

    with (OUT / "crop_quality.csv").open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=report[0].keys()
        )

        writer.writeheader()
        writer.writerows(report)


if comparisons:

    with (OUT / "profile_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=comparisons[0].keys()
        )

        writer.writeheader()
        writer.writerows(comparisons)


# ============================================================
# 10. SAVE GRAYSCALE AND THRESHOLD IMAGES
# ============================================================

for name, (image, gray) in images.items():

    cv2.imwrite(
        str(OUT / f"{name}_gray.png"),
        gray
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    cv2.imwrite(
        str(OUT / f"{name}_otsu.png"),
        binary
    )


# ============================================================
# 11. CREATE CONTACT SHEET
# ============================================================

tiles = []

for name, (image, gray) in images.items():

    scale = min(
        220 / gray.shape[1],
        420 / gray.shape[0]
    )

    nw = max(
        1,
        int(gray.shape[1] * scale)
    )

    nh = max(
        1,
        int(gray.shape[0] * scale)
    )

    enlarged = cv2.resize(
        gray,
        (nw, nh),
        interpolation=cv2.INTER_NEAREST
    )

    tile = np.full(
        (460, 260),
        255,
        dtype=np.uint8
    )

    tile[
        28:28 + nh,
        20:20 + nw
    ] = enlarged

    cv2.putText(
        tile,
        name,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        0,
        1
    )

    tiles.append(tile)


if tiles:

    sheet = np.hstack(tiles)

    cv2.imwrite(
        str(OUT / "crop_contact_sheet.png"),
        sheet
    )


# ============================================================
# 12. FINISHED
# ============================================================

print("\nSaved diagnostics to:", OUT.resolve())
print("Original crops were not changed.")

print("\nGenerated important files:")

for name in profiles:

    if name == "right_main":
        continue

    print(
        f" - alignment_{name}.png"
    )

    print(
        f" - alignment_2d_{name}.png"
    )

    print(
        f" - overlay_2d_{name}.png"
    )

print("\nPHASE 6 DIAGNOSTICS COMPLETE")
