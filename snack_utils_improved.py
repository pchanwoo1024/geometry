# snack_utils_centered.py

import cv2
import numpy as np
from food_data import food_features, nutrition_allergy_db, DAILY_SUGAR_MAX_G, DAILY_SODIUM_MAX_MG

def find_all_contours(gray):
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def pick_central_contour(cnts, img_shape):
    h, w = img_shape[:2]
    cx_img, cy_img = w//2, h//2
    best, best_dist = None, float('inf')
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        dist = (cx-cx_img)**2 + (cy-cy_img)**2
        if dist < best_dist:
            best_dist, best = dist, cnt
    return best

def order_quad(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp_to_front(cnt, img):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx)==4:
        src = approx.reshape(4,2).astype("float32")
    else:
        rect = cv2.minAreaRect(cnt)
        src = cv2.boxPoints(rect).astype("float32")
    src = order_quad(src)
    tl, tr, br, bl = src
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W,H))

def extract_and_classify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnts = find_all_contours(gray)
    if not cnts:
        raise ValueError("컨투어를 찾을 수 없습니다.")
    # ① 중앙에 가장 가까운 컨투어 선택
    central = pick_central_contour(cnts, img.shape)
    warped = warp_to_front(central, img)
    # ② 기하 특성
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cnts2 = find_all_contours(gray_w)
    central2 = pick_central_contour(cnts2, warped.shape)
    x,y,w,h = cv2.boundingRect(central2)
    ratio = max(w,h)/min(w,h)
    area = cv2.contourArea(central2)
    peri = cv2.arcLength(central2, True)
    roundness = (4*np.pi*area/(peri**2)) if peri>0 else 0
    # ③ 색상
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue = int(np.argmax(hist))
    # ④ 분류
    # expected_hue 사전은 이전과 동일
    from food_data import food_features
    expected_hue = {
        "뻥튀기":0, "데미소다":70, "쫀디기":5,
        "메가톤":15,"월드콘":30,"조리퐁":25,
        "미쯔블랙":120,"앙크림빵":20
    }
    best, best_score = None, float('inf')
    for name, feat in food_features.items():
        g = abs(ratio-feat["ratio"]) + abs(roundness-feat["roundness"])
        c = abs(hue-expected_hue[name])/180
        score = g + 0.5*c
        if score<best_score:
            best_score, best = score, name
    # ⑤ 결과
    return best, ratio, roundness, hue, warped, central2

# 예시 사용
# snack, ratio, roundness, hue, warped_img, cnt = extract_and_classify(cv2.imread("test.jpg"))
