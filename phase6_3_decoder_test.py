
import cv2
import zxingcpp
from pyzbar.pyzbar import decode as pyzbar_decode
from pathlib import Path

IMAGE_PATH = Path(
    r"D:\tech\new\phase5_new\10cm0\10cm0_direct_0.png"
)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise SystemExit(f"Cannot load image: {IMAGE_PATH}")

print("Image dimensions:", image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

variants = {
    "original": image,
    "grayscale": gray,
    "upscaled": cv2.resize(
        image, None, fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    ),
    "otsu": cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
}

for name, img in variants.items():

    print("\n---------------------------")
    print("Testing:", name)

    # ZXing
    try:
        results = zxingcpp.read_barcodes(img)

        if results:
            for result in results:
                print(
                    "ZXing:",
                    result.text,
                    "| Format:",
                    result.format
                )
        else:
            print("ZXing: No decode")

    except Exception as e:
        print("ZXing error:", e)

    # Pyzbar
    try:
        results = pyzbar_decode(img)

        if results:
            for result in results:
                print(
                    "Pyzbar:",
                    result.data.decode(
                        "utf-8", errors="replace"
                    ),
                    "| Format:",
                    result.type
                )
        else:
            print("Pyzbar: No decode")

    except Exception as e:
        print("Pyzbar error:", e)
