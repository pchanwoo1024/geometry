# app.py

import streamlit as st
import cv2
import numpy as np

# ─── 1) 스낵 데이터 및 영양·알레르기 정보 ─────────────────────────────
food_classes = {
    "뻥튀기":    {"ratio": 1.00, "roundness": 0.95, "hue": 0,  "taper": 1.0},
    "데미소다":  {"ratio": 3.50, "roundness": 0.90, "hue": 30, "taper": 1.0},
    "쫀디기":    {"ratio": 5.00, "roundness": 0.20, "hue": 10, "taper": 1.0},
    "메가톤":    {"ratio": 3.50, "roundness": 0.30, "hue": 15, "taper": 1.0},
    "월드콘":    {"ratio": 4.00, "roundness": 0.25, "hue": 30, "taper": 10.0},
    "조리퐁":    {"ratio": 1.00, "roundness": 0.65, "hue": 25, "taper": 1.0},
    "미쯔블랙":  {"ratio": 1.00, "roundness": 0.40, "hue":120, "taper": 1.0},
    "앙크림빵":  {"ratio": 1.10, "roundness": 0.90, "hue": 20, "taper": 1.0},
}
nutrition_db = {
    "뻥튀기":    {"열량":100,"당":1,  "나트륨":50,  "알레르기":"없음"},
    "데미소다":  {"열량":140,"당":35, "나트륨":10,  "알레르기":"없음"},
    "쫀디기":    {"열량":150,"당":20, "나트륨":80,  "알레르기":"밀"},
    "메가톤":    {"열량":220,"당":18, "나트륨":150, "알레르기":"우유, 대두, 밀"},
    "월드콘":    {"열량":200,"당":22, "나트륨":120, "알레르기":"우유, 밀"},
    "조리퐁":    {"열량":140,"당":10, "나트륨":100, "알레르기":"밀"},
    "미쯔블랙":  {"열량":180,"당":12, "나트륨":90,  "알레르기":"우유, 밀"},
    "앙크림빵":  {"열량":250,"당":15, "나트륨":180, "알레르기":"우유, 계란, 밀, 대두"},
}
DAILY_SUGAR_MAX  = 50    # g
DAILY_SODIUM_MAX = 2000  # mg
TAPER_MAX        = 15.0  # 콘 분류용 최대 예상 taper

# ─── 2) 전경(라벨) 컨투어 검출 ──────────────────────────────────────
def find_label_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 230)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = img.shape[:2]
    # 전체 테두리 컨투어(90% 이상 면적)는 제거
    return [c for c in cnts if cv2.contourArea(c) < 0.9*(h*w)]

def pick_center_contour(cnts, shape):
    cx_img, cy_img = shape[1]//2, shape[0]//2
    best,bd = None,1e9
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"]==0: continue
        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
        d = (cx-cx_img)**2 + (cy-cy_img)**2
        if d < bd:
            bd,best = d,c
    return best

# ─── 3) 호모그래피로 라벨 정사영 ─────────────────────────────────
def rectify_label(cnt, img):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx)==4:
        src = approx.reshape(4,2).astype("float32")
    else:
        rect = cv2.minAreaRect(cnt)
        src = cv2.boxPoints(rect).astype("float32")
    s = src.sum(axis=1); diff = np.diff(src, axis=1)
    tl = src[np.argmin(s)]; br = src[np.argmax(s)]
    tr = src[np.argmin(diff)]; bl = src[np.argmax(diff)]
    src_pts = np.array([tl,tr,br,bl], dtype="float32")
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst_pts = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (W,H))

# ─── 4) 특성 추출 (ratio, roundness, hue, taper) ──────────────────────
def extract_features(img):
    cnts = find_label_contours(img)
    if not cnts:
        raise ValueError("라벨 영역이 검출되지 않았습니다.")
    cnt0 = pick_center_contour(cnts, img.shape)
    warped = rectify_label(cnt0, img)

    # 기하 특성
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts2,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = pick_center_contour(cnts2, warped.shape)
    x,y,w,h = cv2.boundingRect(cnt1)
    ratio = max(w,h)/min(w,h)
    area  = cv2.contourArea(cnt1)
    peri  = cv2.arcLength(cnt1, True)
    roundness = (4*np.pi*area/(peri**2)) if peri>0 else 0

    # 색상 Hue
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue  = int(np.argmax(hist))

    # 테이퍼 비율 (콘 구분용)
    mask = cv2.inRange(hsv, (20,50,50),(40,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
                            iterations=2)
    w_top = np.count_nonzero(mask[0,:]>0)
    w_bot = np.count_nonzero(mask[-1,:]>0)
    taper = w_top / (w_bot + 1e-3)

    return warped, cnt1, ratio, roundness, hue, taper

# ─── 5) Nearest‐Neighbor 분류 ─────────────────────────────────────
def classify_snack(ratio, roundness, hue, taper):
    # 정규화 및 가중치
    feat = np.array([ratio, roundness, hue/180.0, taper/TAPER_MAX])
    w    = np.array([1.0, 1.0, 0.5, 2.0])  # 조정 가능
    best,bd = None,1e9
    for name, ref in food_classes.items():
        ref_vec = np.array([
            ref["ratio"],
            ref["roundness"],
            ref["hue"]/180.0,
            ref["taper"]/TAPER_MAX
        ])
        dist = np.sqrt(np.sum(w * (feat - ref_vec)**2))
        if dist < bd:
            bd,best = dist,name
    return best

# ─── 6) Streamlit UI ────────────────────────────────────────────
st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너")
st.caption("캔 라벨을 펼쳐서 8개 스낵을 정확히 분류·영양 안내까지!")

up = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if up:
    data = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    try:
        warped, cnt, ratio, roundness, hue, taper = extract_features(img)
        snack  = classify_snack(ratio, roundness, hue, taper)
        info   = nutrition_db[snack]
        msugar = DAILY_SUGAR_MAX  // info["당"]
        msodium= DAILY_SODIUM_MAX // info["나트륨"]

        out = warped.copy()
        cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.success(f"✅ {snack} 인식 완료!")
        st.markdown(f"- 비율: `{ratio:.2f}`  원형도: `{roundness:.2f}`  Hue: `{hue}`  Taper: `{taper:.2f}`")
        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)
        st.markdown("#### ⚠️ 하루 최대 권장 섭취 개수")
        st.write(f"- 당 기준: **{int(msugar)}개**")
        st.write(f"- 나트륨 기준: **{int(msodium)}개**")

    except Exception as e:
        st.error(f"분석 실패: {e}")
