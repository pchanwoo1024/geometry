# app.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# ─── 1) 스낵 클래스 & 영양·알레르기 데이터 ─────────────────────────
food_classes = {
    "뻥튀기":    {"ratio":1.00, "roundness":0.95, "hue":  0, "taper":1.0},
    "데미소다":  {"ratio":3.50, "roundness":0.90, "hue": 30, "taper":1.0},
    "쫀디기":    {"ratio":5.00, "roundness":0.20, "hue": 10, "taper":1.0},
    "메가톤":    {"ratio":3.50, "roundness":0.30, "hue": 15, "taper":1.0},
    "월드콘":    {"ratio":4.00, "roundness":0.25, "hue": 30, "taper":10.0},
    "조리퐁":    {"ratio":1.00, "roundness":0.65, "hue": 25, "taper":1.0},
    "미쯔블랙":  {"ratio":1.00, "roundness":0.40, "hue":120, "taper":1.0},
    "앙크림빵":  {"ratio":1.10, "roundness":0.90, "hue": 20, "taper":1.0},
}
nutrition_db = {
    "뻥튀기":   {"열량":100, "당":1,   "나트륨":50,  "알레르기":"없음"},
    "데미소다": {"열량":140, "당":35,  "나트륨":10,  "알레르기":"없음"},
    "쫀디기":   {"열량":150, "당":20,  "나트륨":80,  "알레르기":"밀"},
    "메가톤":   {"열량":220, "당":18,  "나트륨":150, "알레르기":"우유, 대두, 밀"},
    "월드콘":   {"열량":200, "당":22,  "나트륨":120, "알레르기":"우유, 밀"},
    "조리퐁":   {"열량":140, "당":10,  "나트륨":100, "알레르기":"밀"},
    "미쯔블랙": {"열량":180, "당":12,  "나트륨":90,  "알레르기":"우유, 밀"},
    "앙크림빵": {"열량":250, "당":15,  "나트륨":180, "알레르기":"우유, 계란, 밀, 대두"},
}
DAILY_SUGAR_MAX   = 50    # g
DAILY_SODIUM_MAX  = 2000  # mg
TAPER_MAX         = 15.0  # 콘 분류용 최대 예상 taper

# ─── 2) YOLO‑Seg 모델 로드 ────────────────────────────────────────
_seg_model = YOLO("yolov8n-seg.pt")

def detect_snack_mask(img):
    """YOLOv8 Segmentation으로 스낵 영역 마스크 반환."""
    res = _seg_model(img, verbose=False)[0]
    if not hasattr(res, "masks") or res.masks.data.shape[0] == 0:
        return np.ones(img.shape[:2], dtype=np.uint8) * 255
    masks = res.masks.data.cpu().numpy()  # (N, H, W)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    mask = masks[np.argmax(areas)].astype(np.uint8) * 255
    return mask

# ─── 3) 특징 추출 ────────────────────────────────────────────────
def extract_features(img):
    # A) 세그먼트 → 크롭
    mask = detect_snack_mask(img)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("스낵 영역 분할 실패")
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = img[y1:y2+1, x1:x2+1]

    # B) 라벨 컨투어 검출
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("라벨 윤곽 검출 실패")
    cnt = max(cnts, key=cv2.contourArea)

    # C) 라벨 정사영 (homography)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect).astype("float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    src = np.array([tl, tr, br, bl], dtype="float32")
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(crop, M, (W, H))

    # D) geometry
    gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, th2 = cv2.threshold(gray2, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts2, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = max(cnts2, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt2)
    ratio = max(w, h) / min(w, h)
    area  = cv2.contourArea(cnt2)
    peri2 = cv2.arcLength(cnt2, True)
    roundness = (4 * np.pi * area / (peri2**2)) if peri2>0 else 0

    # E) hue
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
    hue  = int(np.argmax(hist))

    # F) taper
    mask_y = cv2.inRange(hsv, (20,50,50), (40,255,255))
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE,
               cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
               iterations=2)
    w_top = np.count_nonzero(mask_y[0, :] > 0)
    w_bot = np.count_nonzero(mask_y[-1, :] > 0)
    taper = w_top / (w_bot + 1e-3)

    return warped, cnt2, ratio, roundness, hue, taper

# ─── 4) 4D NN 분류 ──────────────────────────────────────────────
def classify_snack(ratio, roundness, hue, taper):
    feat = np.array([ratio, roundness, hue/180.0, taper/TAPER_MAX])
    wgt  = np.array([1.0, 1.0, 0.5, 2.0])
    best, bd = None, float('inf')
    for name, ref in food_classes.items():
        vec = np.array([ref["ratio"],
                        ref["roundness"],
                        ref["hue"]/180.0,
                        ref["taper"]/TAPER_MAX])
        dist = np.sqrt(np.sum(wgt * (feat - vec)**2))
        if dist < bd:
            bd, best = dist, name
    return best

# ─── 5) 전체 파이프라인 ─────────────────────────────────────────
def analyze_snack_image(img):
    warped, cnt, r, rd, hue, taper = extract_features(img)
    snack = classify_snack(r, rd, hue, taper)
    info  = nutrition_db[snack]
    ms    = DAILY_SUGAR_MAX  // info["당"]
    mn    = DAILY_SODIUM_MAX // info["나트륨"]
    out   = warped.copy()
    cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
    return snack, r, rd, hue, taper, info, int(ms), int(mn), out

# ─── 6) Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너 (YOLO‑Seg + 기하분류)")
st.caption("YOLO‑Segmentation → 라벨 정사영 → 8개 스낵 분류·영양 안내까지!")

uploaded = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)

    try:
        snack, ratio, roundness, hue, taper, info, max_sug, max_sod, out = analyze_snack_image(img)

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                 use_container_width=True)
        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 비율: `{ratio:.2f}`  원형도: `{roundness:.2f}`")
        st.markdown(f"- Hue: `{hue}`  Taper: `{taper:.2f}`")
        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)
        st.markdown("#### ⚠️ 하루 최대 권장 섭취 개수")
        st.write(f"- 당 기준: **{max_sug}개**")
        st.write(f"- 나트륨 기준: **{max_sod}개**")

    except Exception as e:
        st.error("분석에 실패했습니다. 상세 오류는 아래 참조:")
        st.exception(e)
