import cv2
import numpy as np
from food_data import food_features, nutrition_allergy_db


def find_largest_contour(gray):
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return max(cnts, key=cv2.contourArea) if cnts else None


def order_quad(pts):
    # tl, tr, br, bl ordering
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect


def warp_to_front(cnt, img):
    # Approximate polygon to get four corners
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        src = approx.reshape(4,2).astype("float32")
    else:
        rect = cv2.minAreaRect(cnt)
        src = cv2.boxPoints(rect).astype("float32")
    src = order_quad(src)
    (tl, tr, br, bl) = src
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    W = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    H = int(max(heightA, heightB))
    dst = np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W, H))


def extract_geometry_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnt = find_largest_contour(gray)
    if cnt is None:
        return None, None, None, None, None
    warped = warp_to_front(cnt, img)
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cnt2 = find_largest_contour(gray_w)
    if cnt2 is None:
        return warped, None, None, None, None
    x, y, w, h = cv2.boundingRect(cnt2)
    ratio = max(w, h) / min(w, h)
    M = cv2.moments(cnt2)
    cx = int(M["m10"] / M["m00"]);
    cy = int(M["m01"] / M["m00"]);
    # shape metrics
    area = cv2.contourArea(cnt2)
    peri = cv2.arcLength(cnt2, True)
    roundness = (4 * np.pi * area / (peri**2)) if peri>0 else 0
    # dominant hue
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
    hue = int(np.argmax(hist))
    return warped, ratio, roundness, (cx, cy), hue


def classify_snack(ratio, roundness, centroid, hue):
    # expected hue map for snacks
    expected_hue = {
        "뻥튀기": 0,
        "초록매실": 70,
        "쫀디기": 5,
        "메가톤": 15,
        "월드콘": 30,
        "조리퐁": 25,
        "미쯔블랙": 120,
        "앙크림빵": 20
    }
    best, best_score = None, float('inf')
    for name, feat in food_features.items():
        g = abs(ratio - feat["ratio"]) + abs(roundness - feat["roundness"])
        c = abs(hue - expected_hue.get(name, hue)) / 180
        score = g + 0.5 * c
        if score < best_score:
            best_score, best = score, name
    return best


def analyze_snack_image(img):
    warped, ratio, roundness, centroid, hue = extract_geometry_features(img)
    if warped is None:
        raise ValueError("평면 패치 검출 실패")
    snack = classify_snack(ratio, roundness, centroid, hue)
    # draw contour for visualization
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cnt2 = find_largest_contour(gray_w)
    out = warped.copy()
    if cnt2 is not None:
        cv2.drawContours(out, [cnt2], -1, (0,255,0), 2)
    info = nutrition_allergy_db.get(snack, {})
    return snack, ratio, roundness, centroid, hue, info, out
