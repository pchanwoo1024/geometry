# snack_utils.py

import cv2
import numpy as np
from ultralytics import YOLO
from food_data import food_classes, nutrition_db, DAILY_SUGAR_MAX, DAILY_SODIUM_MAX, TAPER_MAX

# 1) YOLO 객체검출 모델 로드 (yolov8n는 자동 다운로드)
_model = YOLO("yolov8n.pt")

def detect_snack_bbox(img):
    """YOLO로 사진 속 스낵 바운딩박스를 하나만 리턴."""
    results = _model(img, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    if len(boxes) == 0:
        return None
    idx = np.argmax(confs)
    x1,y1,x2,y2 = boxes[idx]
    return int(x1), int(y1), int(x2), int(y2)

def rectify_label(cnt, img):
    """사각형 근사 + 호모그래피로 라벨 정사영."""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx)==4:
        pts = approx.reshape(4,2).astype("float32")
    else:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect).astype("float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    src = np.array([tl,tr,br,bl], dtype="float32")
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W,H))

def extract_features(img):
    """검출→크롭→정사영→geometry+hue+taper 추출"""
    # A) Detect and crop
    bbox = detect_snack_bbox(img)
    if bbox is None:
        raise ValueError("스낵이 검출되지 않았습니다.")
    x1,y1,x2,y2 = bbox
    crop = img[y1:y2, x1:x2]

    # B) 라벨 컨투어 검출
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("라벨 윤곽이 없습니다.")
    # pick largest
    cnt = max(cnts, key=cv2.contourArea)

    # C) rectify label
    warped = rectify_label(cnt, crop)

    # D) geometry
    gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _,th2 = cv2.threshold(gray2,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts2,_ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = max(cnts2, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt2)
    ratio = max(w,h)/min(w,h)
    area  = cv2.contourArea(cnt2)
    peri  = cv2.arcLength(cnt2, True)
    roundness = (4*np.pi*area/(peri**2)) if peri>0 else 0

    # E) hue
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue  = int(np.argmax(hist))

    # F) taper
    mask = cv2.inRange(hsv, (20,50,50),(40,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
             cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=2)
    w_top = np.count_nonzero(mask[0,:]>0)
    w_bot = np.count_nonzero(mask[-1,:]>0)
    taper = w_top/(w_bot+1e-3)

    return warped, cnt2, ratio, roundness, hue, taper

def classify_snack(ratio, roundness, hue, taper):
    """4D Nearest‐Neighbor 분류"""
    feat = np.array([ratio,
                     roundness,
                     hue/180.0,
                     taper/TAPER_MAX])
    weights = np.array([1.0,1.0,0.5,2.0])
    best,bd = None,1e9
    for name, ref in food_classes.items():
        vec = np.array([ref["ratio"],
                        ref["roundness"],
                        ref["hue"]/180.0,
                        ref["taper"]/TAPER_MAX])
        dist = np.sqrt(np.sum(weights*(feat-vec)**2))
        if dist<bd:
            bd,best = dist,name
    return best

def analyze_snack_image(img):
    warped, cnt, ratio, roundness, hue, taper = extract_features(img)
    snack = classify_snack(ratio, roundness, hue, taper)
    info  = nutrition_db[snack]
    max_sugar  = DAILY_SUGAR_MAX  // info["당"]
    max_sodium = DAILY_SODIUM_MAX // info["나트륨"]
    # draw
    out = warped.copy()
    cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
    return snack, ratio, roundness, hue, taper, info, max_sugar, max_sodium, out
