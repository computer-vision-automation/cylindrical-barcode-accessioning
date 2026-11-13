import cv2
import numpy as np
import os

from pyzbar.pyzbar import decode as zbar_decode
import zxingcpp

folder = r"D:\tech\new\barcode_detail_diagnostic"
files = ["candidate2_linear.png", "candidate2_cubic.png"]

for filename in files:
    path = os.path.join(folder, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Missing:", path)
        continue

    variants = {
        "gray": img,
        "otsu": cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        "adaptive": cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        )
    }

    print("\n", filename)

    for name, image in variants.items():
        for angle in (0, 90, 270):
            test = image if angle == 0 else cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE if angle == 90
                else cv2.ROTATE_90_COUNTERCLOCKWISE
            )

            z = zxingcpp.read_barcodes(test)
            p = zbar_decode(test)

            if z or p:
                print("SUCCESS:", name, "rotation:", angle)
                for result in z:
                    print(" ZXing:", result.format, result.text)
                for result in p:
                    print(" ZBar:", result.type, result.data.decode(
                        "utf-8", errors="replace"
                    ))

    print("Finished:", filename)

print("\nComparison complete.")
