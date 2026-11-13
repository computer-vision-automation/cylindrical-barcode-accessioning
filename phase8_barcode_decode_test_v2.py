#!/usr/bin/env python3
"""Phase 8 V2: robust, non-destructive barcode decode diagnostics.

The input image is never edited. Variants are written to an output folder and
tried with ZXing-C++, ZBar/pyzbar and OpenCV when those engines are available.
No bars/lines are synthetically reconstructed or inpainted.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def gray_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_quality(gray: np.ndarray) -> dict:
    h, w = gray.shape[:2]
    return {
        "width_px": int(w),
        "height_px": int(h),
        "min_intensity": int(gray.min()),
        "max_intensity": int(gray.max()),
        "mean_intensity": round(float(gray.mean()), 3),
        "standard_deviation": round(float(gray.std()), 3),
        "laplacian_variance_blur_indicator": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
        "note": "Laplacian variance is a relative sharpness indicator, not a pass/fail threshold."
    }


def build_variants(gray: np.ndarray, upscale: int = 3) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {}
    rotations = {
        "r0": gray,
        "r90_cw": cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
        "r180": cv2.rotate(gray, cv2.ROTATE_180),
        "r90_ccw": cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    for rot_name, base in rotations.items():
        variants[f"{rot_name}_gray"] = base
        variants[f"{rot_name}_up{upscale}_cubic"] = cv2.resize(
            base, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
        variants[f"{rot_name}_up{upscale}_lanczos"] = cv2.resize(
            base, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base)
        variants[f"{rot_name}_clahe"] = clahe
        variants[f"{rot_name}_clahe_up{upscale}"] = cv2.resize(
            clahe, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
        blur = cv2.GaussianBlur(base, (0, 0), 1.0)
        sharpen = cv2.addWeighted(base, 1.6, blur, -0.6, 0)
        variants[f"{rot_name}_sharpen"] = sharpen
        variants[f"{rot_name}_otsu"] = cv2.threshold(
            base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        variants[f"{rot_name}_otsu_inverted"] = cv2.bitwise_not(variants[f"{rot_name}_otsu"])
        # Adaptive thresholds with two window/offset settings to accommodate uneven lighting.
        for block, c in ((25, 5), (35, 9)):
            variants[f"{rot_name}_adaptive_{block}_{c}"] = cv2.adaptiveThreshold(
                base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c
            )
    return variants


def _normalise_hits(hits) -> list[dict]:
    out = []
    for hit in hits or []:
        txt = getattr(hit, "text", "")
        fmt = getattr(hit, "format", "unknown")
        if txt:
            out.append({"text": str(txt), "format": str(fmt)})
    return out


def decode_zxing(arr: np.ndarray) -> tuple[list[dict] | None, str | None]:
    try:
        import zxingcpp
    except Exception:
        return None, "not_installed"
    try:
        try:
            hits = zxingcpp.read_barcodes(arr, try_rotate=True, try_downscale=True, try_invert=True)
        except TypeError:
            hits = zxingcpp.read_barcodes(arr)
        return _normalise_hits(hits), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def decode_pyzbar(arr: np.ndarray) -> tuple[list[dict] | None, str | None]:
    try:
        from pyzbar.pyzbar import decode
    except Exception:
        return None, "not_installed"
    try:
        hits = decode(arr)
        return [
            {"text": h.data.decode("utf-8", errors="replace"), "format": str(h.type)}
            for h in hits if h.data
        ], None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def decode_opencv(arr: np.ndarray) -> tuple[list[dict] | None, str | None]:
    if not hasattr(cv2, "barcode") or not hasattr(cv2.barcode, "BarcodeDetector"):
        return None, "barcode_module_unavailable"
    try:
        detector = cv2.barcode.BarcodeDetector()
        # OpenCV versions differ in return signatures; support both common forms.
        result = detector.detectAndDecode(arr)
        decoded = result[0] if isinstance(result, tuple) and result else ""
        if isinstance(decoded, (list, tuple)):
            texts = [str(x) for x in decoded if x]
        else:
            texts = [str(decoded)] if decoded else []
        return [{"text": x, "format": "opencv_format_unspecified"} for x in texts], None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run robust barcode decode diagnostics without modifying the source image.")
    ap.add_argument("image", help="Phase 7 unwrapped image (PNG/JPG)")
    ap.add_argument("--out-dir", default="phase8_decode_tests_v2", help="Output directory")
    ap.add_argument("--upscale", type=int, default=3, choices=(2, 3, 4), help="Diagnostic upscale factor")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.is_file():
        ap.error(f"Image not found: {src}")
    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if raw is None:
        ap.error(f"OpenCV could not read image: {src}")
    gray = gray_image(raw)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(gray, args.upscale)

    engines = [("zxing_cpp", decode_zxing), ("pyzbar_zbar", decode_pyzbar), ("opencv_barcode", decode_opencv)]
    report = {
        "phase": 8,
        "version": "2.0",
        "input_image": str(src.resolve()),
        "source_image_modified": False,
        "image_quality": image_quality(gray),
        "variant_count": len(variants),
        "engines": {},
        "successful_decodes": [],
        "interpretation": "No decode does not prove the barcode is absent or invalid. It may reflect damage, blur, low resolution, crop truncation, symbology, or geometry.",
        "repair_policy": "No missing barcode bars are invented, inpainted, or reconstructed. Such edits could change encoded data."
    }

    print("PHASE 8 V2: ROBUST BARCODE DECODER DIAGNOSTICS")
    print("=" * 68)
    print(f"Input: {src} ({gray.shape[1]} x {gray.shape[0]} px)")
    print(f"Sharpness indicator (Laplacian variance): {report['image_quality']['laplacian_variance_blur_indicator']}")
    print(f"Testing {len(variants)} variants with available engines")

    for engine_name, decoder in engines:
        info = {"status": "available", "tested_variants": 0, "errors": []}
        for variant_name, arr in variants.items():
            hits, err = decoder(arr)
            if hits is None and err == "not_installed":
                info["status"] = "not_installed"
                break
            if hits is None:
                info["status"] = "unavailable"
                info["errors"].append(err)
                break
            info["tested_variants"] += 1
            if err:
                info["errors"].append({"variant": variant_name, "error": err})
            for hit in hits:
                item = {"engine": engine_name, "variant": variant_name, **hit}
                report["successful_decodes"].append(item)
                print(f"DECODED [{engine_name}] {variant_name}: {hit['format']} -> {hit['text']}")
        report["engines"][engine_name] = info
        print(f"{engine_name}: {info['status']} ({info['tested_variants']} variants tested)")

    # Save a compact, useful set of inspection images.
    for name in ("r0_gray", "r0_clahe", "r0_otsu", "r90_cw_gray", "r90_ccw_gray", "r180_gray"):
        cv2.imwrite(str(outdir / f"{name}.png"), variants[name])
    report_path = outdir / "phase8_decode_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 68)
    if report["successful_decodes"]:
        unique = {(x["format"], x["text"]) for x in report["successful_decodes"]}
        print(f"SUCCESS: {len(report['successful_decodes'])} hit(s), {len(unique)} unique result(s).")
    else:
        print("NO DECODER RETURNED A BARCODE VALUE.")
        print("Do not infer that the geometry is wrong from this alone.")
    print(f"Report: {report_path.resolve()}")
    print(f"Diagnostics: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
