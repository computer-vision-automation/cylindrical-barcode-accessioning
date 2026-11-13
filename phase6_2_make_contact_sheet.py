
import cv2
import json
import math
from pathlib import Path

ROOT = Path(r"D:\tech\new\phase6_new\auto_detection")

json_path = ROOT / "mirror_detections.json"

with open(json_path, "r", encoding="utf-8") as f:
    detections = json.load(f)

tiles = []

for item in detections:

    path = ROOT / item["crop"]

    image = cv2.imread(str(path))

    if image is None:
        print("Cannot load:", path)
        continue

    # Resize while preserving aspect ratio.
    h, w = image.shape[:2]

    scale = min(240 / w, 220 / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    tile = 255 * (
        cv2.UMat(260, 280, cv2.CV_8UC3).get()
    )

    x = (280 - new_w) // 2
    y = 28 + (220 - new_h) // 2

    tile[y:y+new_h, x:x+new_w] = image

    label = (
        f"Candidate {item['index']} | "
        f"{item['rotation']} deg"
    )

    cv2.putText(
        tile,
        label,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1
    )

    score_text = f"Score: {item['score']:.3f}"

    cv2.putText(
        tile,
        score_text,
        (8, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1
    )

    tiles.append(tile)

if not tiles:
    raise RuntimeError("No candidate images were loaded.")

columns = 4
rows = math.ceil(len(tiles) / columns)

sheet = 255 * (
    cv2.UMat(
        rows * 260,
        columns * 280,
        cv2.CV_8UC3
    ).get()
)

for i, tile in enumerate(tiles):

    row = i // columns
    col = i % columns

    y = row * 260
    x = col * 280

    sheet[y:y+260, x:x+280] = tile

output = ROOT / "mirror_candidates_contact_sheet.jpg"

cv2.imwrite(str(output), sheet)

print("Contact sheet saved:")
print(output)
print("Candidates displayed:", len(tiles))
