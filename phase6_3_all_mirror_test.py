import cv2
import zxingcpp
from pyzbar.pyzbar import decode
from pathlib import Path

base = Path(r"D:\tech\new\phase5_new")

folders = [
    "7cm+90", "7cm-90", "7cm180", "7cm_ran",
    "10cm+90", "10cm180", "10cmran", "10cm0"
]

for folder in folders:
    path = base / folder / f"{folder}_right_rectified.jpg"
    img = cv2.imread(str(path))

    if img is None:
        print(f"\n{folder}: RIGHT RECTIFIED IMAGE MISSING")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

    print(f"\n========== {folder} ==========")

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = gray
        elif angle == 90:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(gray, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

        enlarged = cv2.resize(rotated, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)
        enhanced = clahe.apply(enlarged)
        _, otsu = cv2.threshold(enhanced, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for label, variant in [
            ("gray", enlarged),
            ("clahe", enhanced),
            ("otsu", otsu),
            ("inverse", 255 - otsu)
        ]:
            for result in zxingcpp.read_barcodes(variant):
                print("ZXing DECODE:", result.text,
                      "angle:", angle, "variant:", label)

            for result in decode(variant):
                print("PYZBAR DECODE:",
                      result.data.decode(errors="replace"),
                      "angle:", angle, "variant:", label)

print("\nALL CONFIGURATIONS TESTED")
