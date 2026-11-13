import cv2
import numpy as np
import zxingcpp
from pyzbar.pyzbar import decode
from pathlib import Path

base = Path(r"D:\tech\new\phase5_new\7cm+90")
files = [
    base / "7cm+90_right_3.png",
    base / "7cm+90_right_1.png",
    base / "7cm+90_left_0.png",
]

for path in files:
    img = cv2.imread(str(path))
    if img is None:
        print("MISSING:", path.name)
        continue

    print("\n==========", path.name, "==========")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enlarged)
        _, otsu = cv2.threshold(enhanced, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        variants = {
            "gray": enlarged,
            "clahe": enhanced,
            "otsu": otsu,
            "inverse": 255 - otsu,
        }

        for name, image in variants.items():
            z = zxingcpp.read_barcodes(image)
            p = decode(image)

            for result in z:
                print(f"ZXing angle={angle} variant={name}: "
                      f"{result.text} ({result.format})")

            for result in p:
                print(f"Pyzbar angle={angle} variant={name}: "
                      f"{result.data.decode(errors='replace')} "
                      f"({result.type})")

print("\nTEST COMPLETE")
