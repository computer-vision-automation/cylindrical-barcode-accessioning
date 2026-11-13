
import cv2
from pathlib import Path

ROOT = Path(r"D:\tech\new\phase6_new\7cm+90_vs_10cm0")

images = {
    "direct_reference": ROOT / "10cm0_direct_0.png",
    "mirror_reference": ROOT / "mirror_barcode_roi.png"
}

for name, path in images.items():

    image = cv2.imread(str(path))

    if image is None:
        print("Cannot load:", path)
        continue

    print("\nSelect the complete barcode region.")
    print("Include all bars and their printed numbers.")
    print("Press ENTER or SPACE to confirm.")
    print("Press C to cancel.")

    roi = cv2.selectROI(
        "Select " + name,
        image,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)

    print("\n", name)
    print("x =", x)
    print("y =", y)
    print("width =", w)
    print("height =", h)
    print("ROI =", (x, y, w, h))

    if w > 0 and h > 0:

        crop = image[y:y+h, x:x+w]

        cv2.imwrite(
            str(ROOT / (name + "_selected.png")),
            crop
        )

print("\nROI selection completed.")
