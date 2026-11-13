
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE = Path(
    r"D:\tech\new\phase5_new\7cm+90\7cm+90_right_rectified.jpg"
)

DEFAULT_OUT = Path(
    r"D:\tech\new\phase6_new\phase6_3_validation"
)


# ---------------------------------------------------------
# 1. ROTATION AND COORDINATE MAPPING
# ---------------------------------------------------------

def rotate_image(image, angle):
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def map_box_to_original(box, angle, original_shape):
    x, y, w, h = map(int, box)
    H, W = original_shape[:2]

    if angle == 0:
        return [x, y, w, h]

    if angle == 90:
        return [y, H - (x + w), h, w]

    if angle == 180:
        return [W - (x + w), H - (y + h), w, h]

    return [W - (y + h), x, h, w]


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    return inter / (
        aw * ah + bw * bh - inter + 1e-9
    )


# ---------------------------------------------------------
# 2. AUTOMATIC BARCODE REGION PROPOSALS
# ---------------------------------------------------------

def detect_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad = cv2.Sobel(
        gray, cv2.CV_32F, 1, 0, ksize=3
    )

    grad = cv2.convertScaleAbs(grad)
    grad = cv2.GaussianBlur(grad, (5, 5), 0)

    _, binary = cv2.threshold(
        grad,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    proposals = []

    kernels = [
        (17, 5),
        (25, 7),
        (35, 9),
        (45, 11)
    ]

    for kw, kh in kernels:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kw, kh)
        )

        closed = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        closed = cv2.dilate(
            closed,
            None,
            iterations=2
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            area = w * h

            if area < 1600:
                continue

            if w < 18 or h < 24:
                continue

            aspect = max(
                w / max(h, 1),
                h / max(w, 1)
            )

            if aspect > 8.5:
                continue

            proposals.append({
                "box": [x, y, w, h],
                "method": "sobel_morphology"
            })

    # Optional OpenCV barcode detector.
    try:
        detector = cv2.barcode_BarcodeDetector()
        result = detector.detect(image)

        if isinstance(result, tuple) and len(result) >= 2:
            points = result[1]

            if points is not None:
                for polygon in np.asarray(points).reshape(-1, 4, 2):

                    x1, y1 = np.floor(
                        polygon.min(axis=0)
                    ).astype(int)

                    x2, y2 = np.ceil(
                        polygon.max(axis=0)
                    ).astype(int)

                    if x2 > x1 and y2 > y1:
                        proposals.append({
                            "box": [
                                x1, y1,
                                x2 - x1,
                                y2 - y1
                            ],
                            "method": "opencv_barcode_detector"
                        })

    except Exception:
        pass

    return proposals


# ---------------------------------------------------------
# 3. BARCODE STRUCTURAL FEATURES
# ---------------------------------------------------------

def structural_features(crop):

    if crop is None or crop.size == 0:
        return {"score": 0.0}

    gray = cv2.cvtColor(
        crop, cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    if h < 12 or w < 12:
        return {"score": 0.0}

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    smooth = cv2.GaussianBlur(
        enhanced, (3, 3), 0
    )

    _, binary = cv2.threshold(
        smooth,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    dark = (binary < 128).astype(np.uint8)

    # Fraction of dark pixels in each column.
    column_darkness = dark.mean(axis=0)

    # A barcode should contain repeated dark/light transitions.
    profile = cv2.GaussianBlur(
        column_darkness.astype(np.float32).reshape(1, -1),
        (1, 0),
        1.0
    ).ravel()

    threshold = max(
        0.12,
        min(0.55, float(np.median(profile) + 0.06))
    )

    profile_binary = profile > threshold

    transitions = int(
        np.count_nonzero(
            profile_binary[1:] != profile_binary[:-1]
        )
    )

    # Dark bars should persist along a significant portion
    # of the barcode height.
    persistent = column_darkness > 0.35

    bar_coverage = float(persistent.mean())

    contrast = float(
        np.clip(
            (
                np.percentile(gray, 90)
                - np.percentile(gray, 10)
            ) / 120.0,
            0,
            1
        )
    )

    # Identify tall connected dark components.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, h // 12))
    )

    vertical = cv2.morphologyEx(
        dark * 255,
        cv2.MORPH_OPEN,
        kernel
    )

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            vertical, 8
        )
    )

    tall_components = 0
    heights = []

    for x, y, cw, ch, area in stats[1:]:

        if ch >= 0.30 * h and area >= 2:
            tall_components += 1
            heights.append(ch / h)

    component_score = np.clip(
        tall_components / 18.0, 0, 1
    )

    height_score = (
        float(np.mean(heights))
        if heights else 0.0
    )

    transition_score = np.clip(
        transitions / 24.0, 0, 1
    )

    score = (
        0.38 * transition_score
        + 0.30 * component_score
        + 0.20 * height_score
        + 0.12 * contrast
    )

    return {
        "score": round(float(score), 4),
        "transitions": transitions,
        "bar_coverage": round(bar_coverage, 4),
        "contrast": round(contrast, 4),
        "tall_components": tall_components,
        "mean_component_height_fraction": round(
            height_score, 4
        )
    }


# ---------------------------------------------------------
# 4. CROP AND DEDUPLICATE CANDIDATES
# ---------------------------------------------------------

def crop_box(image, box, padding=8):

    x, y, w, h = box
    H, W = image.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    return image[y1:y2, x1:x2].copy()


def build_candidates(image):

    proposals = []

    for angle in (0, 90, 180, 270):

        rotated = rotate_image(image, angle)

        detections = detect_regions(rotated)

        for detection in detections:

            original_box = map_box_to_original(
                detection["box"],
                angle,
                image.shape
            )

            x, y, w, h = original_box
            image_area = image.shape[0] * image.shape[1]
            candidate_area = w * h
            # Reject regions occupying more than 20% of the image.
            if candidate_area > 0.20 * image_area:
                continue
            if w < 18 or h < 24:
                continue

            proposals.append({
                "box": original_box,
                "rotation": angle,
                "method": detection["method"]
            })

    # Largest regions first.
    proposals.sort(
        key=lambda p: p["box"][2] * p["box"][3],
        reverse=True
    )

    groups = []

    # Merge overlapping detections from different rotations.
    for proposal in proposals:

        matched = None

        for group in groups:

            if box_iou(
                proposal["box"],
                group["box"]
            ) >= 0.28:

                matched = group
                break

        if matched is None:

            groups.append({
                "box": proposal["box"],
                "observations": [proposal]
            })

        else:

           matched["observations"].append(proposal)

# Keep the original representative box.
# Do not expand it using overlapping proposals.
# This prevents chained overlaps from creating
# enormous false candidate regions.

    candidates = []

    for group in groups:

        crop = crop_box(image, group["box"])

        features = structural_features(crop)

        rotations = sorted(set(
            p["rotation"]
            for p in group["observations"]
        ))

        candidates.append({
            "box": group["box"],
            "crop": crop,
            "features": features,
            "rotations": rotations,
            "observations": group["observations"]
        })

    candidates.sort(
        key=lambda c: (
            len(c["rotations"]),
            c["features"]["score"]
        ),
        reverse=True
    )

    return candidates


# ---------------------------------------------------------
# 5. MULTI-ORIENTATION BARCODE DECODING
# ---------------------------------------------------------

def decode_candidate(crop):

    decoders = []

    try:
        import zxingcpp

        decoders.append((
            "ZXing",
            lambda img: zxingcpp.read_barcodes(img)
        ))

    except ImportError:
        pass

    try:
        from pyzbar.pyzbar import decode as pyzbar_decode

        decoders.append((
            "Pyzbar",
            lambda img: pyzbar_decode(img)
        ))

    except ImportError:
        pass

    try:
        detector = cv2.barcode_BarcodeDetector()

        def opencv_decode(img):
            ok, text, types, points = (
                detector.detectAndDecode(img)
            )
            return [text] if ok and text else []

        decoders.append(("OpenCV", opencv_decode))

    except Exception:
        pass

    results = []
    seen = set()

    for angle in (0, 90, 180, 270):

        oriented = rotate_image(crop, angle)

        gray = cv2.cvtColor(
            oriented, cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            2.0, (8, 8)
        ).apply(gray)

        otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        # CLAHE followed by Otsu thresholding
        clahe_otsu = cv2.threshold(
            clahe,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        # Inverted threshold image
        otsu_inv = cv2.bitwise_not(otsu)
        # Upscaled binary images
        otsu_upscaled = cv2.resize(
            otsu,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
            )
        clahe_otsu_upscaled = cv2.resize(
            clahe_otsu,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
            )
        variants = [
            ("raw", oriented),
            ("clahe", clahe),
            
            ("otsu", otsu),
            ("otsu_inverted", otsu_inv),
            ("clahe_otsu", clahe_otsu),
            ("otsu_upscaled", otsu_upscaled),
            ("clahe_otsu_upscaled", clahe_otsu_upscaled)
            ]

        for variant_name, variant in variants:
            if angle == 0:
                print(
        f"Testing candidate crop: "
        f"{crop.shape[1]}x{crop.shape[0]} | "
        f"Variants: {[v[0] for v in variants]}"
        )

            for decoder_name, decoder in decoders:

                try:
                    decoded = decoder(variant)

                    for result in decoded or []:

                        if isinstance(result, str):

                            text = result
                            fmt = "unknown"

                        else:

                            text = (
                                getattr(result, "text", "")
                                or getattr(
                                    result,
                                    "data",
                                    b""
                                ).decode(
                                    "utf-8",
                                    "replace"
                                )
                            )

                            fmt = str(
                                getattr(
                                    result,
                                    "format",
                                    "unknown"
                                )
                            )

                        if not text:
                            continue
                        print(
                            f"DECODE SUCCESS: {decoder_name} | "
                            f"Rotation: {angle} | "
                            f"Preprocessing: {variant_name} | "
                            f"Text: {text}"
                            )
                        key = (text, fmt)
                        if key not in seen:
                            seen.add(key)
                            results.append({
                                "text": str(text),
                                "format": fmt,
                                "decoder": decoder_name,
                                "rotation": angle,
                                "preprocessing": variant_name
                                })

                except Exception:
                    continue

    return results


# ---------------------------------------------------------
# 6. CONTACT SHEET
# ---------------------------------------------------------

def save_contact_sheet(candidates, output_path):

    if not candidates:
        return

    cols = 4
    tile_w = 260
    tile_h = 190

    rows = math.ceil(len(candidates) / cols)

    sheet = np.zeros(
        (rows * tile_h, cols * tile_w, 3),
        dtype=np.uint8
    )

    for i, candidate in enumerate(candidates):

        crop = candidate["crop"]

        x = (i % cols) * tile_w
        y = (i // cols) * tile_h

        scale = min(
            (tile_w - 12) / crop.shape[1],
            (tile_h - 48) / crop.shape[0]
        )

        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        xx = x + (tile_w - resized.shape[1]) // 2
        yy = y + 28 + (
            tile_h - 48 - resized.shape[0]
        ) // 2

        sheet[
            yy:yy + resized.shape[0],
            xx:xx + resized.shape[1]
        ] = resized

        score = candidate["features"]["score"]

        label = (
            f"#{i:02d} score={score:.3f} "
            f"views={candidate['rotations']}"
        )

        cv2.putText(
            sheet,
            label,
            (x + 5, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (230, 230, 230),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(str(output_path), sheet)


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE)
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT)
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.out)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = cv2.imread(
        str(source_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise SystemExit(
            f"Could not load image: {source_path}"
        )

    print("Source image:", source_path)
    print("Image dimensions:", image.shape)

    candidates = build_candidates(image)

    candidates = candidates[:args.max_candidates]

    print("\nUnique candidates:", len(candidates))

    annotated = image.copy()

    json_results = []

    report = [
        "PHASE 6.3 AUTOMATIC BARCODE VALIDATION",
        f"Source: {source_path}",
        f"Dimensions: {image.shape[1]}x{image.shape[0]}",
        f"Candidates: {len(candidates)}",
        "",
        "Scores are heuristic ranking values, NOT probabilities.",
        ""
    ]

    for i, candidate in enumerate(candidates):

        crop = candidate["crop"]

        crop_name = f"candidate_{i:02d}.png"

        cv2.imwrite(
            str(output_dir / crop_name),
            crop
        )

        decodes = decode_candidate(crop)

        candidate["decode_results"] = decodes

        x, y, w, h = candidate["box"]

        color = (
            (0, 255, 0)
            if decodes
            else (0, 190, 255)
        )

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            annotated,
            f"{i}: {candidate['features']['score']:.2f}",
            (x, max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        decoded_text = (
            sorted(set(d["text"] for d in decodes))
            if decodes else []
        )

        line = (
            f"[{i:02d}] "
            f"box={candidate['box']} "
            f"score={candidate['features']['score']:.4f} "
            f"rotations={candidate['rotations']} "
            f"transitions={candidate['features'].get('transitions', 0)} "
            f"tall_bars={candidate['features'].get('tall_components', 0)} "
            f"decoded={decoded_text or 'NONE'}"
        )

        print(line)
        report.append(line)

        json_results.append({
            "index": i,
            "box_xywh": candidate["box"],
            "crop_file": crop_name,
            "rotations": candidate["rotations"],
            "observations": candidate["observations"],
            "features": candidate["features"],
            "decode_results": decodes
        })

    cv2.imwrite(
        str(output_dir / "annotated_candidates.jpg"),
        annotated
    )

    save_contact_sheet(
        candidates,
        output_dir / "phase6_3_contact_sheet.jpg"
    )

    payload = {
        "phase": "6.3",
        "source": str(source_path),
        "candidate_count": len(json_results),
        "score_note": "Heuristic ranking score, not a probability.",
        "candidates": json_results
    }

    with open(
        output_dir / "phase6_3_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(payload, f, indent=2)

    with open(
        output_dir / "phase6_3_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(report))

    print("\n--------------------------------")
    print("PHASE 6.3 COMPLETE")
    print("--------------------------------")
    print("Output folder:", output_dir)
    print("Contact sheet: phase6_3_contact_sheet.jpg")
    print("Annotated image: annotated_candidates.jpg")
    print("JSON: phase6_3_results.json")
    print("Report: phase6_3_report.txt")


if __name__ == "__main__":
    main()
