
import cv2
import zxingcpp
from pyzbar.pyzbar import decode as pyzbar_decode
from pathlib import Path

BASE = Path(
    r"D:\tech\new\phase6_new\phase6_3_validation"
)

CANDIDATES = [
    "candidate_07.png",
    "candidate_08.png"
]

for filename in CANDIDATES:
    path = BASE / filename
    image = cv2.imread(str(path))

    if image is None:
        print("Could not load:", path)
        continue

    print("\n" + "=" * 50)
    print("Testing:", filename)
    print("Dimensions:", image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    variants = {
        "gray": gray,
        "otsu": cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        "otsu_inverted": cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1],
        "clahe_otsu": cv2.threshold(
            clahe, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
    }

    for scale in (2, 3, 4):
        for name, variant in variants.items():
            enlarged = cv2.resize(
                variant,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            print(f"\n{name}, scale={scale}")

            zxing = zxingcpp.read_barcodes(enlarged)
            print(
                "ZXing:",
                [r.text for r in zxing if r.text] or "NONE"
            )

            pyzbar = pyzbar_decode(enlarged)
            print(
                "Pyzbar:",
                [
                    r.data.decode("utf-8", errors="replace")
                    for r in pyzbar
                ] or "NONE"
            )
