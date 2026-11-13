import cv2
img = cv2.imread(r"samples\7cm180.jpg")
boxes = {"left": (560, 1850, 860, 2200), "right": (2270, 1850, 2550, 2200)}  # x1,y1,x2,y2 in original px
for name, (x1, y1, x2, y2) in boxes.items():
    c = img[y1:y2, x1:x2]
    cv2.imwrite(f"crop180_{name}.png", c)
    cv2.imwrite(f"crop180_{name}_4x.png", cv2.resize(c, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    