import streamlit as st
import cv2
import numpy as np

# ─── 1) 스낵 정보 ──────────────────────────────────────────────
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
    "뻥튀기":    {"열량":100,"탄수":24,"단백질":2,"지방":0.5,"당":1,  "나트륨":50,  "알레르기":"없음"},
    "데미소다":  {"열량":140,"탄수":35,"단백질":0,"지방":0,  "당":35, "나트륨":10,  "알레르기":"없음"},
    "쫀디기":    {"열량":150,"탄수":30,"단백질":3,"지방":1,  "당":20, "나트륨":80,  "알레르기":"밀"},
    "메가톤":    {"열량":220,"탄수":27,"단백질":2.5,"지방":12,"당":18, "나트륨":150, "알레르기":"우유, 대두, 밀"},
    "월드콘":    {"열량":200,"탄수":25,"단백질":3,"지방":10, "당":22, "나트륨":120, "알레르기":"우유, 밀"},
    "조리퐁":    {"열량":140,"탄수":22,"단백질":3,"지방":2,  "당":10, "나트륨":100, "알레르기":"밀"},
    "미쯔블랙": {"열량":180,"탄수":20,"단백질":2,"지방":8,  "당":12, "나트륨":90,  "알레르기":"우유, 밀"},
    "앙크림빵": {"열량":250,"탄수":30,"단백질":5,"지방":8,  "당":15, "나트륨":180, "알레르기":"우유, 계란, 밀, 대두"}
}

DAILY_SUGAR_MAX   = 50    # g
DAILY_SODIUM_MAX  = 2000  # mg

# ─── 2) 라벨 영역 검출 & 정사영 보정 ───────────────────────────
def detect_label_quad(img):
    """HSV에서 노란색 라벨 영역 검출 → 4코너 근사 → homography"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 노란색 범위 (Hue 20~40, Sat>50, Val>50)
    mask = cv2.inRange(hsv, (20,50,50), (40,255,255))
    # 모폴로지로 노이즈 제거/구멍 메우기
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("라벨 영역을 찾을 수 없습니다.")
    # 가장 큰 노란 영역 선택
    label = max(cnts, key=cv2.contourArea)
    # 사각근사
    peri = cv2.arcLength(label, True)
    approx = cv2.approxPolyDP(label, 0.02*peri, True)
    if len(approx) == 4:
        src = approx.reshape(4,2).astype("float32")
    else:
        # fallback: 최소면적 사각
        rect = cv2.minAreaRect(label)
        src = cv2.boxPoints(rect).astype("float32")
    # 4코너 순서 정리 (TL,TR,BR,BL)
    s = src.sum(axis=1)
    diff = np.diff(src, axis=1)
    ordered = np.zeros((4,2), dtype="float32")
    ordered[0] = src[np.argmin(s)]
    ordered[2] = src[np.argmax(s)]
    ordered[1] = src[np.argmin(diff)]
    ordered[3] = src[np.argmax(diff)]
    tl,tr,br,bl = ordered
    # 목표 크기: label bounding box 크기로
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    # homography
    Hmat = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, Hmat, (W,H))
    return warped

# ─── 3) 기하+색상 특징 추출 & 분류 ────────────────────────────
def extract_and_classify(img):
    # 1) label만 정사영 보정
    warped = detect_label_quad(img)
    # 2) 중앙 컨투어(라벨 전체) 검출
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("보정 후 라벨 윤곽을 찾을 수 없습니다.")
    # pick largest
    cnt = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    ratio = max(w,h)/min(w,h)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    roundness = (4*np.pi*area/(peri**2)) if peri>0 else 0
    # 3) hue도 재계산
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue = int(np.argmax(hist))
    # 4) 분류
    expected_hue = {
      "뻥튀기":0, "데미소다":30, "쫀디기":5,
      "메가톤":15,"월드콘":30,"조리퐁":25,
      "미쯔블랙":120,"앙크림빵":20
    }
    best, bs = None, 1e9
    for name, feat in food_features.items():
        g = abs(ratio-feat["ratio"]) + abs(roundness-feat["roundness"])
        c = abs(hue-expected_hue[name])/180.0
        score = g + 0.5*c
        if score<bs:
            bs, best = score, name
    # 5) 영양정보 & 권장량
    info = nutrition_allergy_db[best]
    max_su = DAILY_SUGAR_MAX  // info["당"]
    max_na = DAILY_SODIUM_MAX // info["나트륨"]
    # 6) 윤곽선 시각화
    out = warped.copy()
    cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
    return best, ratio, roundness, hue, info, max_su, max_na, out

# ─── 4) Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너")
st.caption("캔 라벨만 자동으로 펼쳐드립니다. 사진 한가운데 올려보세요!")

up = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if up:
    data = np.asarray(bytearray(up.read()),dtype=np.uint8)
    img  = cv2.imdecode(data,cv2.IMREAD_COLOR)
    try:
        snack, r, rd, hue, info, ms, mn, out = extract_and_classify(img)
        st.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB), use_column_width=True)
        st.success(f"✅ {snack} 인식 완료!")
        st.markdown(f"- 가로/세로 비율: `{r:.2f}`  ")
        st.markdown(f"- 원형도: `{rd:.2f}`  - Hue: `{hue}`")
        st.markdown("#### ℹ️ 영양·알레르기")
        st.table(info)
        st.markdown("#### ⚠️ 최대 권장 섭취 개수")
        st.write(f"- 당 기준: **{ms}개**")
        st.write(f"- 나트륨 기준: **{mn}개**")
    except Exception as e:
        st.error(f"분석 실패: {e}")
