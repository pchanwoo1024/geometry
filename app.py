# app.py

import streamlit as st
import cv2
import numpy as np

# ─── 1) 스낵 클래스 & 영양·알레르기 데이터 ─────────────────────
food_classes = {
    "뻥튀기":   {"ratio":1.00,"roundness":0.95,"hue":0,  "taper":1.0},
    "데미소다": {"ratio":3.50,"roundness":0.90,"hue":30, "taper":1.0},
    "쫀디기":   {"ratio":5.00,"roundness":0.20,"hue":10, "taper":1.0},
    "메가톤":   {"ratio":3.50,"roundness":0.30,"hue":15, "taper":1.0},
    "월드콘":   {"ratio":4.00,"roundness":0.25,"hue":30, "taper":10.0},
    "조리퐁":   {"ratio":1.00,"roundness":0.65,"hue":25, "taper":1.0},
    "미쯔블랙": {"ratio":1.00,"roundness":0.40,"hue":120,"taper":1.0},
    "앙크림빵": {"ratio":1.10,"roundness":0.90,"hue":20, "taper":1.0},
}
nutrition_db = {
    "뻥튀기":   {"열량":100,"당":1,  "나트륨":50,  "알레르기":"없음"},
    "데미소다": {"열량":140,"당":35, "나트륨":10,  "알레르기":"없음"},
    "쫀디기":   {"열량":150,"당":20, "나트륨":80,  "알레르기":"밀"},
    "메가톤":   {"열량":220,"당":18, "나트륨":150, "알레르기":"우유, 대두, 밀"},
    "월드콘":   {"열량":200,"당":22, "나트륨":120, "알레르기":"우유, 밀"},
    "조리퐁":   {"열량":140,"당":10, "나트륨":100, "알레르기":"밀"},
    "미쯔블랙": {"열량":180,"당":12, "나트륨":90,  "알레르기":"우유, 밀"},
    "앙크림빵": {"열량":250,"당":15, "나트륨":180, "알레르기":"우유, 계란, 밀, 대두"},
}
DAILY_SUGAR_MAX  = 50
DAILY_SODIUM_MAX = 2000
TAPER_MAX        = 15.0

# ─── 2) 라벨 사각형 검출: Canny + 4꼭짓점 ─────────────────────
def find_label_quad(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 밝은 배경 제거
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # 에지 검출
    edges = cv2.Canny(th, 50, 150)
    # 모폴로지로 구멍 메우기
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            area = abs(cv2.contourArea(approx))
            if area > best_area:
                best_area, best = area, approx.reshape(4,2)
    if best is None:
        raise ValueError("라벨 사각형을 찾을 수 없습니다.")
    return best.astype("float32")

# ─── 3) 정사영(orthographic) ────────────────────────────────────
def warp_quad(img, quad):
    pts = quad
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    src = np.array([tl,tr,br,bl], dtype="float32")
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W,H))

# ─── 4) 특성 추출 & 분류 ────────────────────────────────────────
def analyze_snack_image(img):
    quad = find_label_quad(img)
    warped = warp_quad(img, quad)

    # geometry
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts2,_ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = max(cnts2, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt2)
    ratio = max(w,h)/min(w,h)
    area  = cv2.contourArea(cnt2)
    peri  = cv2.arcLength(cnt2, True)
    roundness = (4*np.pi*area/(peri**2)) if peri>0 else 0

    # hue
    hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue  = int(np.argmax(hist))

    # taper
    mask = cv2.inRange(hsv, (20,50,50),(40,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
             cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=2)
    w_top = np.count_nonzero(mask[0,:]>0)
    w_bot = np.count_nonzero(mask[-1,:]>0)
    taper = w_top/(w_bot+1e-3)

    # classify
    feat = np.array([ratio, roundness, hue/180.0, taper/TAPER_MAX])
    wgt  = np.array([1.0,1.0,0.5,2.0])
    best,bd = None,1e9
    for name, ref in food_classes.items():
        vec = np.array([ref["ratio"],ref["roundness"],ref["hue"]/180.0,ref["taper"]/TAPER_MAX])
        d = np.sqrt(np.sum(wgt*(feat-vec)**2))
        if d<bd: bd,best=d,name

    info = nutrition_db[best]
    ms   = DAILY_SUGAR_MAX  // info["당"]
    mn   = DAILY_SODIUM_MAX // info["나트륨"]

    out = warped.copy()
    cv2.drawContours(out, [cnt2], -1, (0,255,0),2)
    return best, ratio, roundness, hue, taper, info, int(ms), int(mn), out

# ─── 5) Streamlit UI ────────────────────────────────────────────
st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너 (Contour‑Quad + 기하분류)")
st.caption("Canny 에지 → 4점 다각형 검출 → 정사영 → 8개 스낵 분류·영양 안내까지!")

up = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if up:
    data = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)

    try:
        snack, ratio, roundness, hue, taper, info, ms, mn, out = analyze_snack_image(img)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 비율: `{ratio:.2f}`, 원형도: `{roundness:.2f}`")
        st.markdown(f"- Hue: `{hue}`, Taper: `{taper:.2f}`")
        st.markdown("#### ℹ️ 영양·알레르기")
        st.table(info)
        st.markdown("#### ⚠️ 권장 최대 섭취 개수")
        st.write(f"- 당 기준: {ms}개  |  나트륨 기준: {mn}개")
    except Exception as e:
        st.error("분석 실패:")
        st.exception(e)
