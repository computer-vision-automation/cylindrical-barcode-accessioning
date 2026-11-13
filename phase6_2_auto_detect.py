
import cv2
import numpy as np
import json
from pathlib import Path

ROOT = Path(r"D:\tech\new")

INPUTS = {
    "direct": ROOT / "phase5_new" / "10cm0" /
              "10cm0_direct_0.png",

    "mirror": ROOT / "phase5_new" / "7cm+90" /
              "7cm+90_right_rectified.jpg"
}

OUT = ROOT / "phase6_new" / "auto_detection"

OUT.mkdir(parents=True, exist_ok=True)


def rotate_image(image, angle):
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def barcode_detector_candidates(image):
    results = []

    try:
        detector = cv2.barcode.BarcodeDetector()
        ok, decoded_info, decoded_type, points = (
            detector.detectAndDecodeMulti(image)
        )

        if ok and points is not None:
            for polygon in points:
                polygon = np.asarray(polygon).reshape(-1, 2)
                x, y, w, h = cv2.boundingRect(
                    polygon.astype(np.int32)
                )
                results.append((x, y, w, h, "opencv_barcode"))

    except (AttributeError, cv2.error):
        pass

    return results


def morphology_candidates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    candidates = []

    # Test both parallel-bar orientations.
    for orientation in ("horizontal", "vertical"):

        if orientation == "horizontal":
            grad = cv2.Sobel(
                gray, cv2.CV_32F, 0, 1, ksize=3
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (21, 3)
            )
        else:
            grad = cv2.Sobel(
                gray, cv2.CV_32F, 1, 0, ksize=3
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (3, 21)
            )

        grad = np.abs(grad)

        grad = cv2.normalize(
            grad, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        grad = cv2.GaussianBlur(grad, (3, 3), 0)

        _, binary = cv2.threshold(
            grad, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        connected = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel
        )

        # Join nearby bar edges into candidate regions.
        if orientation == "horizontal":
            join = cv2.getStructuringElement(
                cv2.MORPH_RECT, (9, 17)
            )
        else:
            join = cv2.getStructuringElement(
                cv2.MORPH_RECT, (17, 9)
            )

        connected = cv2.dilate(
            connected, join, iterations=2
        )

        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        image_area = image.shape[0] * image.shape[1]

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            area = w * h

            if area < image_area * 0.0005:
                continue

            if w < 18 or h < 30:
                continue

            if w > image.shape[1] * 0.9:
                continue

            if h > image.shape[0] * 0.9:
                continue

            roi = gray[y:y+h, x:x+w]

            # Measure repeated edge transitions.
            if orientation == "horizontal":
                projection = np.mean(
                    cv2.Sobel(
                        roi, cv2.CV_32F, 0, 1, ksize=3
                    ), axis=1
                )
            else:
                projection = np.mean(
                    cv2.Sobel(
                        roi, cv2.CV_32F, 1, 0, ksize=3
                    ), axis=0
                )

            projection = np.abs(projection)

            if len(projection) < 10:
                continue

            threshold = np.mean(projection) + np.std(projection)

            active = projection > threshold

            transitions = np.count_nonzero(
                np.diff(active.astype(np.uint8))
            )

            # Heuristic score, NOT a probability.
            density = transitions / max(len(projection), 1)

            score = (
                density *
                min(area / (image_area * 0.02), 1.0)
            )

            candidates.append({
                "box": (x, y, w, h),
                "score": float(score),
                "orientation": orientation,
                "method": "morphology"
            })

    return candidates


def extract_candidates(name, original):

    all_candidates = []

    for angle in (0, 90, 180, 270):

        image = rotate_image(original, angle)

        # Try native OpenCV barcode detector.
        for x, y, w, h, method in barcode_detector_candidates(image):
            all_candidates.append({
                "box": (x, y, w, h),
                "score": 2.0,
                "orientation": "detector",
                "rotation": angle,
                "method": method
            })

        # Try morphology-based proposals.
        for item in morphology_candidates(image):
            item["rotation"] = angle
            all_candidates.append(item)

    all_candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Suppress highly overlapping boxes.
    kept = []

    for candidate in all_candidates:

        x, y, w, h = candidate["box"]

        duplicate = False

        for previous in kept:

            px, py, pw, ph = previous["box"]

            ix1 = max(x, px)
            iy1 = max(y, py)
            ix2 = min(x+w, px+pw)
            iy2 = min(y+h, py+ph)

            intersection = max(0, ix2-ix1) * max(0, iy2-iy1)

            union = w*h + pw*ph - intersection

            iou = intersection / max(union, 1)

            if iou > 0.65:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept[:20]



def process(name, path):

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load image: {path}"
        )

    print(f"\nProcessing {name}: {image.shape}")

    # Direct reference is already a Phase 5 barcode crop.
    if name == "direct":

        crop_name = "direct_reference.png"

        cv2.imwrite(
            str(OUT / crop_name),
            image
        )

        report = [{
            "index": 0,
            "box": [
                0, 0,
                image.shape[1],
                image.shape[0]
            ],
            "score": None,
            "orientation": "existing_phase5_crop",
            "rotation": 0,
            "method": "phase5_reference",
            "crop": crop_name
        }]

        cv2.imwrite(
            str(OUT / "direct_all_candidates.jpg"),
            image
        )

        with open(
            OUT / "direct_detections.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(report, f, indent=4)

        print("Using existing Phase 5 reference crop.")
        return

    # IMPORTANT:
    # extract_candidates already checks all four rotations.
    # Therefore, call it only ONCE.
    candidates = extract_candidates(
        name,
        image
    )

    # Suppress duplicates within the same rotation.
    kept = []

    for candidate in candidates:

        x, y, w, h = candidate["box"]

        duplicate = False

        for previous in kept:

            if candidate["rotation"] != previous["rotation"]:
                continue

            px, py, pw, ph = previous["box"]

            ix1 = max(x, px)
            iy1 = max(y, py)
            ix2 = min(x+w, px+pw)
            iy2 = min(y+h, py+ph)

            intersection = (
                max(0, ix2-ix1) *
                max(0, iy2-iy1)
            )

            union = (
                w*h + pw*ph - intersection
            )

            iou = intersection / max(union, 1)

            if iou > 0.65:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    kept = kept[:30]

    report = []
    annotated_images = {}

    for index, candidate in enumerate(kept):

        angle = candidate["rotation"]

        # Recreate the exact image in which this
        # candidate's coordinates were measured.
        rotated = rotate_image(image, angle)

        if angle not in annotated_images:
            annotated_images[angle] = rotated.copy()

        x, y, w, h = candidate["box"]

        h_img, w_img = rotated.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x+w)
        y2 = min(h_img, y+h)

        if x1 >= x2 or y1 >= y2:
            print("Skipping invalid candidate:", index)
            continue

        crop = rotated[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_name = f"mirror_candidate_{index}.png"

        if not cv2.imwrite(
            str(OUT / crop_name),
            crop
        ):
            print("Could not save:", crop_name)
            continue

        cv2.rectangle(
            annotated_images[angle],
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_images[angle],
            f"{index}: {candidate['score']:.3f}",
            (x1, max(20, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        report.append({
            "index": index,
            "box": [x1, y1, x2-x1, y2-y1],
            "score": candidate["score"],
            "orientation": candidate["orientation"],
            "rotation": angle,
            "method": candidate["method"],
            "crop": crop_name
        })

    for angle, annotated in annotated_images.items():

        cv2.imwrite(
            str(OUT / f"mirror_rotation_{angle}.jpg"),
            annotated
        )

    with open(
        OUT / "mirror_detections.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(report, f, indent=4)

    print("\nMirror candidates:", len(report))

    for item in report[:10]:

        print(
            item["index"],
            item["box"],
            "rotation:", item["rotation"],
            "score:", round(item["score"], 3)
        )

def main():

    for name, path in INPUTS.items():
        process(name, path)

    print("\nAutomated detection completed.")
    print("Results:", OUT)


if __name__ == "__main__":
    main()
