# app.py
import streamlit as st
import cv2
import numpy as np

# ─── 1) 스낵 데이터 ────────────────────────────────────────────
food_features = {
    "뻥튀기":    {"ratio":1.0, "roundness":0.95},
    "데미소다":  {"ratio":3.5, "roundness":0.90},
    "쫀디기":    {"ratio":5.0, "roundness":0.20},
    "메가톤":    {"ratio":3.5, "roundness":0.30},
    "월드콘":    {"ratio":4.0, "roundness":0.25},
    "조리퐁":    {"ratio":1.0, "roundness":0.65},
    "미쯔블랙": {"ratio":1.0, "roundness":0.40},
    "앙크림빵": {"ratio":1.1, "roundness":0.90}
}
nutrition_allergy_db = {
    "뻥튀기":    {"열량(kcal)":100,"탄수화물(g)":24,"단백질(g)":2, "지방(g)":0.5, "당(g)":1,  "나트륨(mg)":50,  "알레르기":"없음"},
    "데미소다":  {"열량(kcal)":140,"탄수화물(g)":35,"단백질(g)":0, "지방(g)":0,   "당(g)":35, "나트륨(mg)":10,  "알레르기":"없음"},
    "쫀디기":    {"열량(kcal)":150,"탄수화물(g)":30,"단백질(g)":3, "지방(g)":1,   "당(g)":20, "나트륨(mg)":80,  "알레르기":"밀"},
    "메가톤":    {"열량(kcal)":220,"탄수화물(g)":27,"단백질(g)":2.5, "지방(g)":12, "당(g)":18, "나트륨(mg)":150, "알레르기":"우유, 대두, 밀"},
    "월드콘":    {"열량(kcal)":200,"탄수화물(g)":25,"단백질(g)":3,   "지방(g)":10,  "당(g)":22, "나트륨(mg)":120, "알레르기":"우유, 밀"},
    "조리퐁":    {"열량(kcal)":140,"탄수화물(g)":22,"단백질(g)":3,   "지방(g)":2,   "당(g)":10, "나트륨(mg)":100, "알레르기":"밀"},
    "미쯔블랙": {"열량(kcal)":180,"탄수화물(g)":20,"단백질(g)":2,   "지방(g)":8,   "당(g)":12, "나트륨(mg)":90,  "알레르기":"우유, 밀"},
    "앙크림빵": {"열량(kcal)":250,"탄수화물(g)":30,"단백질(g)":5,   "지방(g)":8,   "당(g)":15, "나트륨(mg)":180, "알레르기":"우유, 계란, 밀, 대두"}
}
DAILY_SUGAR_MAX   = 50    # g
DAILY_SODIUM_MAX  = 2000  # mg

# ─── 2) 배경 제거 + 중앙 컨투어 선택 ───────────────────────────
def find_label_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 밝은 배경 제거
    mask = cv2.inRange(gray, 0, 230)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 테두리 필터링
    h,w = img.shape[:2]
    cnts = [c for c in cnts if cv2.contourArea(c) < 0.9*(h*w)]
    return cnts

def pick_center_contour(cnts, shape):
    cx_img, cy_img = shape[1]//2, shape[0]//2
    best,bd = None,1e9
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"]==0: continue
        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
        d = (cx-cx_img)**2 + (cy-cy_img)**2
        if d<bd:
            bd,best = d,c
    return best

# ─── 3) 사각근사 & 호모그래피 보정 ───────────────────────────
def rectify_label(cnt, img):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx)==4:
        src = approx.reshape(4,2).astype("float32")
    else:
        rect = cv2.minAreaRect(cnt)
        src = cv2.boxPoints(rect).astype("float32")
    # TL,TR,BR,BL 정렬
    s = src.sum(axis=1); diff = np.diff(src,axis=1)
    tl = src[np.argmin(s)]; br = src[np.argmax(s)]
    tr = src[np.argmin(diff)]; bl = src[np.argmax(diff)]
    src_pts = np.array([tl,tr,br,bl], dtype="float32")
    # 출력 크기
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst_pts = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (W,H))

# ─── 4) 특징 추출 & 분류 ──────────────────────────────────────
def extract_features(img):
    cnts = find_label_contours(img)
    if not cnts:
        raise ValueError("스낵 라벨이 검출되지 않았습니다.")
    center_cnt = pick_center_contour(cnts, img.shape)
    warped = rectify_label(center_cnt, img)
    # 다시 컨투어 추출
    cnts2 = find_label_contours(warped)
    cnt2  = pick_center_contour(cnts2, warped.shape)
    x,y,w,h = cv2.boundingRect(cnt2)
    ratio = max(w,h)/min(w,h)
    area  = cv2.contourArea(cnt2)
    peri  = cv2.arcLength(cnt2,True)
    roundness = (4*np.pi*area/(peri**2)) if peri>0 else 0
    # Hue
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue  = int(np.argmax(hist))
    return warped, cnt2, ratio, roundness, hue

def classify_snack(ratio, roundness, hue):
    # 기대 색상
    exp_hue = {"뻥튀기":0,"데미소다":30,"쫀디기":5,
               "메가톤":15,"월드콘":30,"조리퐁":25,
               "미쯔블랙":120,"앙크림빵":20}
    best,bs = None,1e9
    for name,feat in food_features.items():
        g = abs(ratio-feat["ratio"]) + abs(roundness-feat["roundness"])
        c = abs(hue-exp_hue[name])/180.0
        score = g + 0.5*c
        if score<bs: bs,best = score,name
    return best

# ─── 5) Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너")
st.caption("가운데 스낵 라벨만 펼치고 분류합니다. 사진 중앙에 올려보세요!")

uploaded = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if uploaded:
    data = np.asarray(bytearray(uploaded.read()),dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)

    try:
        warped, cnt_label, ratio, roundness, hue = extract_features(img)
        snack = classify_snack(ratio, roundness, hue)
        info  = nutrition_allergy_db[snack]
        # 권장 섭취 개수
        ms = DAILY_SUGAR_MAX  // info["당(g)"]   if info["당(g)"]>0   else float('inf')
        mn = DAILY_SODIUM_MAX // info["나트륨(mg)"] if info["나트륨(mg)"]>0 else float('inf')

        # 시각화
        out = warped.copy()
        cv2.drawContours(out, [cnt_label], -1, (0,255,0), 2)
        st.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB), use_column_width=True)

        st.success(f"✅ {snack} 인식 완료!")
        st.markdown(f"- 비율: `{ratio:.2f}`  원형도: `{roundness:.2f}`  Hue: `{hue}`")
        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)
        st.markdown("#### ⚠️ 권장 최대 섭취 개수")
        st.write(f"- 당 기준: **{int(ms)}개**")
        st.write(f"- 나트륨 기준: **{int(mn)}개**")

    except Exception as e:
        st.error(f"분석 실패: {e}")
