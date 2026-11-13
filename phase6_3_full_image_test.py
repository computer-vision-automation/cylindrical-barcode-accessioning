
import cv2
import zxingcpp
from pyzbar.pyzbar import decode as pyzbar_decode
from pathlib import Path

SOURCE = Path(
    r"D:\tech\new\phase5_new\7cm+90\7cm+90_right_rectified.jpg"
)

image = cv2.imread(str(SOURCE))

if image is None:
    raise SystemExit(f"Cannot load image: {SOURCE}")

print("Image dimensions:", image.shape)

def rotate(img, angle):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

found = set()

for angle in (0, 90, 180, 270):
    oriented = rotate(image, angle)

    gray = cv2.cvtColor(
        oriented, cv2.COLOR_BGR2GRAY
    )

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
        )[1]
    }

    for scale in (1, 2):
        for name, variant in variants.items():

            if scale == 2:
                variant = cv2.resize(
                    variant,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

            for decoder_name, decoder in (
                ("ZXing", zxingcpp.read_barcodes),
                ("Pyzbar", pyzbar_decode)
            ):
                try:
                    results = decoder(variant)

                    for result in results:
                        if isinstance(result, str):
                            text = result
                        else:
                            text = (
                                getattr(result, "text", "")
                                or getattr(result, "data", b"").decode(
                                    "utf-8", "replace"
                                )
                            )

                        if text:
                            key = (
                                decoder_name,
                                text,
                                angle,
                                name,
                                scale
                            )

                            if key not in found:
                                found.add(key)

                                print(
                                    "DECODED:",
                                    decoder_name,
                                    "| Text:", text,
                                    "| Rotation:", angle,
                                    "| Variant:", name,
                                    "| Scale:", scale
                                )

                except Exception as exc:
                    print("Decoder error:", decoder_name, exc)

if not found:
    print("\nNO BARCODE DECODED IN THE FULL MIRROR IMAGE.")

print("\nFull-image test complete.")
