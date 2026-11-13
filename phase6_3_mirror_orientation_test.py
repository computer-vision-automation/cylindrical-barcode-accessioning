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

success = False

for path in files:
    img = cv2.imread(str(path))
    if img is None:
        print("MISSING:", path.name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for flip_name, source in [
        ("original", gray),
        ("horizontal_flip", cv2.flip(gray, 1))
    ]:
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = source
            elif angle == 90:
                rotated = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(source, cv2.ROTATE_180)
            else:
                rotated = cv2.rotate(source, cv2.ROTATE_90_COUNTERCLOCKWISE)

            enlarged = cv2.resize(
                rotated, None, fx=4, fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
            enhanced = clahe.apply(enlarged)
            _, otsu = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            variants = {
                "gray": enlarged,
                "clahe": enhanced,
                "otsu": otsu,
                "inverse": 255 - otsu
            }

            for variant_name, image in variants.items():
                for result in zxingcpp.read_barcodes(image):
                    print(
                        "SUCCESS ZXing:",
                        path.name, result.text,
                        flip_name, angle, variant_name
                    )
                    success = True

                for result in decode(image):
                    print(
                        "SUCCESS Pyzbar:",
                        path.name,
                        result.data.decode(errors="replace"),
                        flip_name, angle, variant_name
                    )
                    success = True

if not success:
    print("NO DECODE FOUND IN ANY ORIENTATION.")

print("ORIENTATION TEST COMPLETE")
