import cv2
import numpy as np
from ultralytics import YOLO
from food_data import food_classes, nutrition_db, DAILY_SUGAR_MAX, DAILY_SODIUM_MAX, TAPER_MAX

# YOLOv8 Segmentation 모델 로드
_seg_model = YOLO("yolov8n-seg.pt")

def detect_snack_mask(img):
    """YOLOv8 Segmentation으로 스낵 인스턴스 마스크 반환."""
    res = _seg_model(img, verbose=False)[0]
    if not hasattr(res, "masks") or res.masks.data.shape[0] == 0:
        # fallback: 전체 이미지 마스크
        return np.ones(img.shape[:2], dtype=np.uint8)*255
    # 가장 큰 면적 마스크 선택
    masks = res.masks.data.cpu().numpy()  # (N, H, W)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    m = masks[np.argmax(areas)].astype(np.uint8)*255
    return m

def extract_features(img):
    """Seg → 크롭 → 라벨 정사영 → 4D 특징 추출"""
    # 1) 세그먼트 마스크로 정확히 스낵 영역만 크롭
    mask = detect_snack_mask(img)
    # bounding box from mask
    ys, xs = np.where(mask>0)
    if len(xs)==0:
        raise ValueError("세그먼트 실패: 마스크가 없습니다.")
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = img[y1:y2+1, x1:x2+1]

    # 이하 기존대로 'crop'에서 라벨 contour → 정사영 → 특징 추출
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, 
                         cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, 
                              cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("라벨 윤곽 검출 실패")
    cnt = max(cnts, key=cv2.contourArea)

    # label rectify
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx)==4:
        pts = approx.reshape(4,2).astype("float32")
    else:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect).astype("float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    src = np.array([tl,tr,br,bl], dtype="float32")
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(crop, M, (W,H))

    # geometry
    gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, th2 = cv2.threshold(gray2, 0,255,
                          cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts2,_ = cv2.findContours(th2, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = max(cnts2, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt2)
    ratio = max(w,h)/min(w,h)
    area  = cv2.contourArea(cnt2)
    peri2 = cv2.arcLength(cnt2, True)
    roundness = (4*np.pi*area/(peri2**2)) if peri2>0 else 0

    # hue
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue  = int(np.argmax(hist))

    # taper
    mask_yellow = cv2.inRange(hsv, (20,50,50),(40,255,255))
    mask_yellow = cv2.morphologyEx(mask_yellow,
        cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_RECT,(5,5)), iterations=2)
    w_top = np.count_nonzero(mask_yellow[0   ,:]>0)
    w_bot = np.count_nonzero(mask_yellow[-1  ,:]>0)
    taper = w_top/(w_bot+1e-3)

    return warped, cnt2, ratio, roundness, hue, taper

# classify_snack & analyze_snack_image 는 그대로 사용
