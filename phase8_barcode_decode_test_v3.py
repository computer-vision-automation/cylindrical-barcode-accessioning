
#!/usr/bin/env python3
"""
PHASE 8 V3: Barcode ROI isolation and decoder diagnostics.

- Preserves the original image.
- Detects likely horizontal barcode bars.
- Creates a padded barcode ROI.
- Tests full image and ROI in multiple orientations.
- Uses ZXing-C++, Pyzbar and OpenCV.
- Separates detection from successful decoding.
- Never invents or reconstructs missing barcode bars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def to_gray(image):
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_quality(gray):
    return {
        "width_px": int(gray.shape[1]),
        "height_px": int(gray.shape[0]),
        "min_intensity": int(gray.min()),
        "max_intensity": int(gray.max()),
        "mean_intensity": round(float(gray.mean()), 3),
        "standard_deviation": round(float(gray.std()), 3),
        "laplacian_variance": round(
            float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3
        )
    }


def find_barcode_roi(gray):
    """
    Detect long horizontal dark structures.
    Intended for this project's horizontal-bar / vertical-sequence image.
    """

    h, w = gray.shape

    otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    kernel_width = max(12, int(w * 0.22))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 1)
    )

    boxes = []

    for mask in (otsu, adaptive):
        horizontal = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )

        horizontal = cv2.dilate(
            horizontal,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (3, 3)
            ),
            iterations=1
        )

        contours, _ = cv2.findContours(
            horizontal,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)

            if bw >= max(18, int(w * 0.30)):
                boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        return None, {
            "found": False,
            "reason": "No sufficiently long horizontal bars detected."
        }

    # Combine the detected bar fragments into one barcode region.
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    # Add padding to avoid clipping barcode edges.
    pad_x = max(4, int((x2 - x1) * 0.10))
    pad_y = max(5, int((y2 - y1) * 0.04))

    xa = max(0, x1 - pad_x)
    ya = max(0, y1 - pad_y)
    xb = min(w, x2 + pad_x)
    yb = min(h, y2 + pad_y)

    roi = gray[ya:yb, xa:xb].copy()

    return roi, {
        "found": True,
        "bbox_xywh": [
            int(xa), int(ya),
            int(xb - xa), int(yb - ya)
        ],
        "raw_bar_bbox_xywh": [
            int(x1), int(y1),
            int(x2 - x1), int(y2 - y1)
        ],
        "padding_px": [pad_x, pad_y],
        "detected_bar_fragments": len(boxes),
        "method": "Otsu/adaptive threshold + horizontal morphology",
        "warning": (
            "Automatic ROI is a proposal. Inspect it for clipped bars "
            "and quiet zones before trusting the result."
        )
    }


def build_variants(gray, prefix, upscale):
    variants = {}

    rotations = {
        "r0": gray,
        "r90_cw": cv2.rotate(
            gray, cv2.ROTATE_90_CLOCKWISE
        ),
        "r180": cv2.rotate(
            gray, cv2.ROTATE_180
        ),
        "r90_ccw": cv2.rotate(
            gray, cv2.ROTATE_90_COUNTERCLOCKWISE
        )
    }

    for rotation_name, base in rotations.items():

        name = f"{prefix}_{rotation_name}"

        variants[f"{name}_gray"] = base

        variants[f"{name}_upscale"] = cv2.resize(
            base, None,
            fx=upscale, fy=upscale,
            interpolation=cv2.INTER_CUBIC
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(base)

        variants[f"{name}_clahe"] = clahe

        blurred = cv2.GaussianBlur(
            clahe, (0, 0), 1.0
        )

        variants[f"{name}_sharpen"] = cv2.addWeighted(
            clahe, 1.5, blurred, -0.5, 0
        )

        variants[f"{name}_otsu"] = cv2.threshold(
            clahe, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        variants[f"{name}_otsu_inverted"] = cv2.bitwise_not(
            variants[f"{name}_otsu"]
        )

        for block, c in ((25, 5), (35, 9)):
            if min(base.shape) > block:
                variants[f"{name}_adaptive_{block}"] = (
                    cv2.adaptiveThreshold(
                        clahe, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block, c
                    )
                )

    return variants


def decode_zxing(image):
    try:
        import zxingcpp
    except Exception:
        return None, "not_installed", {}

    try:
        try:
            hits = zxingcpp.read_barcodes(
                image,
                try_rotate=True,
                try_downscale=True,
                try_invert=True
            )
        except TypeError:
            hits = zxingcpp.read_barcodes(image)

        results = []

        for hit in hits or []:
            text_value = getattr(hit, "text", "")

            if text_value:
                results.append({
                    "text": str(text_value),
                    "format": str(
                        getattr(hit, "format", "unknown")
                    )
                })

        return results, None, {
            "decoder_returned_hits": len(results)
        }

    except Exception as exc:
        return [], str(exc), {}


def decode_pyzbar(image):
    try:
        from pyzbar.pyzbar import decode
    except Exception:
        return None, "not_installed", {}

    try:
        hits = decode(image)
        results = []

        for hit in hits:
            if hit.data:
                results.append({
                    "text": hit.data.decode(
                        "utf-8", errors="replace"
                    ),
                    "format": str(hit.type),
                    "quality": int(hit.quality),
                    "rectangle": [
                        int(hit.rect.left),
                        int(hit.rect.top),
                        int(hit.rect.width),
                        int(hit.rect.height)
                    ]
                })

        return results, None, {
            "decoder_returned_hits": len(results)
        }

    except Exception as exc:
        return [], str(exc), {}


def decode_opencv(image):
    """
    Handle different OpenCV Python return signatures.

    Common signature:
    retval, decoded_info, decoded_type, points
    """

    if not hasattr(cv2, "barcode"):
        return None, "barcode_module_unavailable", {}

    if not hasattr(cv2.barcode, "BarcodeDetector"):
        return None, "barcode_detector_unavailable", {}

    try:
        detector = cv2.barcode.BarcodeDetector()
        result = detector.detectAndDecode(image)

        decoded_strings = []
        points_available = False

        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, str) and item:
                    decoded_strings.append(item)

                elif isinstance(item, (list, tuple)):
                    for value in item:
                        if isinstance(value, str) and value:
                            decoded_strings.append(value)

                elif isinstance(item, np.ndarray):
                    if item.size > 0:
                        points_available = True

        elif isinstance(result, str) and result:
            decoded_strings.append(result)

        # If decoding failed, check whether detection alone works.
        detected = False

        try:
            detection_result = detector.detect(image)

            if isinstance(detection_result, tuple):
                for item in detection_result:
                    if isinstance(item, (bool, np.bool_)):
                        detected = detected or bool(item)

                    elif isinstance(item, np.ndarray):
                        if item.size > 0:
                            points_available = True
                            detected = True

        except Exception:
            pass

        results = [
            {
                "text": value,
                "format": "opencv_format_unspecified"
            }
            for value in sorted(set(decoded_strings))
        ]

        return results, None, {
            "detected_without_decode": detected and not results,
            "points_available": points_available,
            "decoded_value_count": len(results)
        }

    except Exception as exc:
        return [], str(exc), {}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 V3 barcode ROI and decoder diagnostics."
    )

    parser.add_argument("image")
    parser.add_argument(
        "--out-dir",
        default="phase8_decode_tests_v3"
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=3,
        choices=(2, 3, 4)
    )

    args = parser.parse_args()

    source = Path(args.image)

    if not source.is_file():
        parser.error(f"Image not found: {source}")

    raw = cv2.imread(
        str(source), cv2.IMREAD_UNCHANGED
    )

    if raw is None:
        parser.error("Could not read input image.")

    gray = to_gray(raw)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    roi, roi_info = find_barcode_roi(gray)

    # Always test the complete image.
    variants = build_variants(
        gray, "full", args.upscale
    )

    # Also test the isolated candidate ROI.
    if roi is not None:
        cv2.imwrite(
            str(output / "candidate_barcode_roi.png"),
            roi
        )

        roi_variants = build_variants(
            roi, "roi", args.upscale
        )

        variants.update(roi_variants)

    engines = {
        "zxing_cpp": decode_zxing,
        "pyzbar_zbar": decode_pyzbar,
        "opencv_barcode": decode_opencv
    }

    report = {
        "phase": 8,
        "version": "3.0",
        "input_image": str(source.resolve()),
        "source_image_modified": False,
        "image_quality": image_quality(gray),
        "roi_detection": roi_info,
        "variant_count": len(variants),
        "engines": {},
        "successful_decodes": [],
        "interpretation": (
            "A failed decode does not establish that the barcode "
            "is damaged or invalid."
        ),
        "repair_policy": (
            "No missing barcode bars are invented, inpainted "
            "or reconstructed."
        )
    }

    print("\nPHASE 8 V3: ROI + BARCODE DECODER DIAGNOSTICS")
    print("=" * 68)

    print(
        f"Input: {source.name} "
        f"({gray.shape[1]} x {gray.shape[0]} px)"
    )

    print("ROI detection:", roi_info["found"])

    if roi_info["found"]:
        print("ROI bbox:", roi_info["bbox_xywh"])

    print("Total variants:", len(variants))

    for engine_name, decoder in engines.items():

        info = {
            "status": "available",
            "tested_variants": 0,
            "variants_with_decoded_values": 0,
            "detection_without_decode": [],
            "errors": []
        }

        for variant_name, image in variants.items():

            hits, error, extra = decoder(image)

            if hits is None:
                info["status"] = "not_installed"
                break

            info["tested_variants"] += 1

            if error:
                info["errors"].append({
                    "variant": variant_name,
                    "error": error
                })

            if extra.get("detected_without_decode"):
                info["detection_without_decode"].append(
                    variant_name
                )

            if hits:
                info["variants_with_decoded_values"] += 1

            for hit in hits:
                record = {
                    "engine": engine_name,
                    "variant": variant_name,
                    **hit
                }

                report["successful_decodes"].append(record)

                print(
                    f"\nDECODED: {engine_name} | "
                    f"{variant_name} | "
                    f"{hit['format']} | {hit['text']}"
                )

        report["engines"][engine_name] = info

        print(
            f"{engine_name}: {info['status']} | "
            f"{info['tested_variants']} tested | "
            f"{info['variants_with_decoded_values']} decoded"
        )

        if info["detection_without_decode"]:
            print(
                "  Detection without decoding:",
                len(info["detection_without_decode"])
            )

    # Save useful diagnostic images.
    for name in (
        "full_r0_gray",
        "full_r0_clahe",
        "full_r0_otsu",
        "full_r90_cw_gray",
        "full_r90_ccw_gray"
    ):
        if name in variants:
            cv2.imwrite(
                str(output / f"{name}.png"),
                variants[name]
            )

    if roi is not None:
        for name in (
            "roi_r0_gray",
            "roi_r0_clahe",
            "roi_r0_otsu",
            "roi_r90_cw_gray",
            "roi_r90_ccw_gray"
        ):
            if name in variants:
                cv2.imwrite(
                    str(output / f"{name}.png"),
                    variants[name]
                )

    report_path = output / "phase8_decode_report_v3.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 68)

    if report["successful_decodes"]:
        print(
            "SUCCESS:",
            len(report["successful_decodes"]),
            "decoded results"
        )
    else:
        print("NO DECODER RETURNED A BARCODE VALUE.")

    print("Report:", report_path.resolve())
    print("Diagnostics:", output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
