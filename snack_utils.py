import cv2
import numpy as np
from food_data import food_features, nutrition_allergy_db

def find_largest_contour(gray):
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None

def warp_to_front(cnt, img):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    w, h = int(rect[1][0]), int(rect[1][1])
    dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, H, (w,h))
    return warped

def extract_geometry_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnt = find_largest_contour(gray)
    if cnt is None: return None, None, None
    warped = warp_to_front(cnt, img)
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cnt2 = find_largest_contour(gray_w)
    if cnt2 is None: return None, None, None
    x,y,w,h = cv2.boundingRect(cnt2)
    ratio = max(w,h) / min(w,h)
    area = cv2.contourArea(cnt2)
    peri = cv2.arcLength(cnt2, True)
    roundness = (4 * np.pi * area / (peri**2)) if peri>0 else 0
    return warped, ratio, roundness

def classify_snack(ratio, roundness):
    best, best_score = None, float('inf')
    for name, feat in food_features.items():
        score = abs(ratio - feat["ratio"]) + abs(roundness - feat["roundness"])
        if score < best_score:
            best_score, best = score, name
    return best

def analyze_snack_image(img):
    warped, ratio, roundness = extract_geometry_features(img)
    if warped is None:
        raise ValueError("기하 분석 실패")
    snack = classify_snack(ratio, roundness)
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cnt2 = find_largest_contour(gray_w)
    out = warped.copy()
    if cnt2 is not None:
        cv2.drawContours(out, [cnt2], -1, (0,255,0), 2)
    return snack, ratio, roundness, nutrition_allergy_db[snack], out