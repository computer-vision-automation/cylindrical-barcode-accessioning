import cv2
import zxingcpp
from pyzbar.pyzbar import decode
from pathlib import Path

base = Path(r"D:\tech\new\phase5_new")

files = [
    base / "10cm+90" / "10cm+90_right_1.png",
    base / "10cm180" / "10cm180_right_3.png",
    base / "10cmran" / "10cmran_right_1.png",
    base / "7cm+90" / "7cm+90_right_1.png",
    base / "7cm+90" / "7cm+90_right_2.png",
    base / "7cm+90" / "7cm+90_right_3.png",
]

for path in files:
    img = cv2.imread(str(path))
    if img is None:
        print("\nMISSING:", path)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

    print("\n==========", path.name, "==========")

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = gray
        elif angle == 90:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(gray, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

        enlarged = cv2.resize(rotated, None, fx=4, fy=4,
                              interpolation=cv2.INTER_CUBIC)
        enhanced = clahe.apply(enlarged)
        _, otsu = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        for label, variant in [
            ("gray", enlarged),
            ("clahe", enhanced),
            ("otsu", otsu),
            ("inverse", 255 - otsu)
        ]:
            for result in zxingcpp.read_barcodes(variant):
                print("ZXing:", result.text, result.format,
                      "angle:", angle, "variant:", label)

            for result in decode(variant):
                print("Pyzbar:", result.data.decode(errors="replace"),
                      result.type, "angle:", angle, "variant:", label)

print("\nBATCH TEST COMPLETE")
