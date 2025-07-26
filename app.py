# app.py
import streamlit as st
import cv2, pytesseract
import numpy as np

# ─── 0) OCR 설정 ─────────────────────────────────────────────
# (Tesseract가 PATH에 있을 경우 불필요)
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

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
nutrition = {
  "뻥튀기":    {"열량":100,"당":1,"나트륨":50,"알레르기":"없음"},
  "데미소다":  {"열량":140,"당":35,"나트륨":10,"알레르기":"없음"},
  "쫀디기":    {"열량":150,"당":20,"나트륨":80,"알레르기":"밀"},
  "메가톤":    {"열량":220,"당":18,"나트륨":150,"알레르기":"우유, 대두, 밀"},
  "월드콘":    {"열량":200,"당":22,"나트륨":120,"알레르기":"우유, 밀"},
  "조리퐁":    {"열량":140,"당":10,"나트륨":100,"알레르기":"밀"},
  "미쯔블랙": {"열량":180,"당":12,"나트륨":90,"알레르기":"우유, 밀"},
  "앙크림빵": {"열량":250,"당":15,"나트륨":180,"알레르기":"우유, 계란, 밀, 대두"}
}
SUGAR_MAX, SODIUM_MAX = 50, 2000

# ─── 2) 유틸 함수 ──────────────────────────────────────────────
def pick_center(cnts, shape):
    cx, cy = shape[1]//2, shape[0]//2
    best, bd = None, 1e9
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"]==0: continue
        x, y = M["m10"]/M["m00"], M["m01"]/M["m00"]
        d = (x-cx)**2 + (y-cy)**2
        if d<bd: bd, best = d, c
    return best

def order_quad(pts):
    rect = np.zeros((4,2),dtype="float32")
    s = pts.sum(axis=1); rect[0],rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts,axis=1); rect[1],rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp_quad(img, cnt):
    peri = cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,0.02*peri,True)
    pts = approx.reshape(4,2) if len(approx)==4 else cv2.boxPoints(cv2.minAreaRect(cnt))
    src = order_quad(pts.astype("float32"))
    tl,tr,br,bl = src
    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W,H))

# ─── 3) 메인 로직 ─────────────────────────────────────────────
def analyze(img):
    # 1) 노란색 라벨 마스크
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20,50,50),(40,255,255))
    # 노이즈 제거
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    # 2) 중앙 라벨만
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = pick_center(cnts, img.shape)
    warped = warp_quad(img, cnt)

    # 3) OCR -> 텍스트 매핑
    gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="kor+eng").strip()
    # 예: '데미소다' 단어 포함 검사
    for name in food_features:
        if name in text:
            return name, warped, cnt

    # 4) fallback: 기하+색상 분류
    # (기존 비율/원형도+hue 분류 로직 삽입...)
    # 생략: classify_snack(ratio,roundness,hue) 같은 방식

    return "미확인", warped, cnt

# ─── 4) Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="푸드스캐너",layout="centered")
st.title("📷 푸드스캐너 (OCR 보강판)")
up = st.file_uploader("사진 업로드", type=["jpg","png","jpeg"])
if up:
    data = np.asarray(bytearray(up.read()),dtype=np.uint8)
    img  = cv2.imdecode(data,cv2.IMREAD_COLOR)
    try:
        name, w, cnt = analyze(img)
        out = w.copy(); cv2.drawContours(out,[cnt],-1,(0,255,0),2)
        st.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),use_column_width=True)
        st.success(f"==> {name} 인식!")
        info = nutrition[name]
        st.table(info)
    except Exception as e:
        st.error("인식 실패: " + str(e))
