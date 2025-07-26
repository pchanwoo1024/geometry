import cv2
import numpy as np
from ultralytics import YOLO
from food_data import food_classes, nutrition_db, DAILY_SUGAR_MAX, DAILY_SODIUM_MAX, TAPER_MAX

# YOLOv8 객체검출 모델 로드 (최초 실행 시 자동 다운로드)
_model = YOLO("yolov8n.pt")

def detect_snack_bbox(img):
    """YOLO로 스낵 검출, 실패 시 전체 이미지 반환."""
    results = _model(img, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    if len(boxes) == 0:
        h, w = img.shape[:2]
        return 0, 0, w, h
    idx = np.argmax(confs)
    x1,y1,x2,y2 = boxes[idx]
    return int(x1), int(y1), int(x2), int(y2)

# 이하 라벨 정사영·4D 특징 추출·분류 함수는 이전과 동일
# ...
# (extract_features, classify_snack, analyze_snack_image)
