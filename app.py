import streamlit as st
import cv2
import numpy as np

# ─── 1) 스낵 데이터 ────────────────────────────────────────────
food_features = {
    "뻥튀기":    {"ratio":1.0, "roundness":0.95},
    "데미소다":  {"ratio":1.0, "roundness":0.95},
    "쫀디기":    {"ratio":5.0, "roundness":0.20},
    "메가톤":    {"ratio":3.5, "roundness":0.30},
    "월드콘":    {"ratio":4.0, "roundness":0.25},
    "조리퐁":    {"ratio":1.0, "roundness":0.65},
    "미쯔블랙": {"ratio":1.0, "roundness":0.40},
    "앙크림빵": {"ratio":1.1, "roundness":0.90}
}

nutrition_allergy_db = {
    "뻥튀기":    {"열량(kcal)":100,"탄수(g)":24,"단백질(g)":2,"지방(g)":0.5,"당(g)":1,"나트륨(mg)":50,"알레르기":"없음"},
    "데미소다":  {"열량(kcal)":140,"탄수(g)":35,"단백질(g)":0,"지방(g)":0,"당(g)":35,"나트륨(mg)":10,"알레르기":"없음"},
    "쫀디기":    {"열량(kcal)":150,"탄수(g)":30,"단백질(g)":3,"지방(g)":1,"당(g)":20,"나트륨(mg)":80,"알레르기":"밀"},
    "메가톤":    {"열량(kcal)":220,"탄수(g)":27,"단백질(g)":2.5,"지방(g)":12,"당(g)":18,"나트륨(mg)":150,"알레르기":"우유, 대두, 밀"},
    "월드콘":    {"열량(kcal)":200,"탄수(g)":25,"단백질(g)":3,"지방(g)":10,"당(g)":22,"나트륨(mg)":120,"알레르기":"우유, 밀"},
    "조리퐁":    {"열량(kcal)":140,"탄수(g)":22,"단백질(g)":3,"지방(g)":2,"당(g)":10,"나트륨(mg)":100,"알레르기":"밀"},
    "미쯔블랙": {"열량(kcal)":180,"탄수(g)":20,"단백질(g)":2,"지방(g)":8,"당(g)":12,"나트륨(mg)":90,"알레르기":"우유, 밀"},
    "앙크림빵": {"열량(kcal)":250,"탄수(g)":30,"단백질(g)":5,"지 지방(g)":8,"당(g)":15,"나트륨(mg)":180,"알레르기":"우유, 계란, 밀, 대두"}
}

DAILY_SUGAR_MAX = 50    # g
DAILY_SODIUM_MAX = 2000 # mg

# ─── 2) 이미지 처리 유틸 ──────────────────────────────────────
def find_all_contours(gray):
    blur = cv2.GaussianBlur(gray,(7,7),0)
    _,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    closed = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel)
    cnts,_ = cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def pick_central_contour(cnts, shape):
    h,w = shape[:2]
    cx_img, cy_img = w//2, h//2
    best,bd = None,1e9
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"]==0: continue
        cx = int(M["m10"]/M["m00"]); cy=int(M["m01"]/M["m00"])
        d=(cx-cx_img)**2+(cy-cy_img)**2
        if d<bd: bd,d, best = d,d,c
    return best

def order_quad(pts):
    rect = np.zeros((4,2),dtype="float32")
    s = pts.sum(axis=1)
    rect[0],rect[2]=pts[np.argmin(s)],pts[np.argmax(s)]
    d = np.diff(pts,axis=1)
    rect[1],rect[3]=pts[np.argmin(d)],pts[np.argmax(d)]
    return rect

def warp_to_front(cnt,img):
    peri = cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,0.02*peri,True)
    if len(approx)==4:
        src=approx.reshape(4,2).astype("float32")
    else:
        rect=cv2.minAreaRect(cnt)
        src=cv2.boxPoints(rect).astype("float32")
    src=order_quad(src)
    tl,tr,br,bl=src
    W=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    H=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img,M,(W,H))

def extract_features(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cnts=find_all_contours(gray)
    if not cnts: raise ValueError("컨투어없음")
    central=pick_central_contour(cnts,img.shape)
    warped=warp_to_front(central,img)
    # 다시 컨투어
    gray2=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    cnts2=find_all_contours(gray2)
    central2=pick_central_contour(cnts2,warped.shape)
    x,y,w,h=cv2.boundingRect(central2)
    ratio=max(w,h)/min(w,h)
    area=cv2.contourArea(central2)
    peri=cv2.arcLength(central2,True)
    roundness=(4*np.pi*area/(peri**2)) if peri>0 else 0
    hsv=cv2.cvtColor(warped,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0],None,[180],[0,180])
    hue=int(np.argmax(hist))
    return warped,ratio,roundness,hue,central2

def classify_snack(ratio,roundness,hue):
    expected_hue={"뻥튀기":0,"데미소다":70,"쫀디기":5,
                  "메가톤":15,"월드콘":30,"조리퐁":25,
                  "미쯔블랙":120,"앙크림빵":20}
    best,bs=None,1e9
    for name,feat in food_features.items():
        g=abs(ratio-feat["ratio"])+abs(roundness-feat["roundness"])
        c=abs(hue-expected_hue[name])/180
        score=g+0.5*c
        if score<bs: bs, best=score,name
    return best

# ─── 3) Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="푸드스캐너",layout="centered")
st.title("📷 푸드스캐너")
st.caption("사진 올리면 스낵 인식→영양·알레르기·안전 권장량까지!")

up = st.file_uploader("스낵 사진을 올려주세요",type=["jpg","png","jpeg"])
if up:
    data=np.asarray(bytearray(up.read()),dtype=np.uint8)
    img=cv2.imdecode(data,cv2.IMREAD_COLOR)
    try:
        warped,ratio,roundness,hue,cnt=extract_features(img)
        snack=classify_snack(ratio,roundness,hue)
        info=nutrition_allergy_db[snack]
        sugar, sodium = info["당(g)"], info["나트륨(mg)"]
        max_su=DAILY_SUGAR_MAX//sugar if sugar>0 else float('inf')
        max_na=DAILY_SODIUM_MAX//sodium if sodium>0 else float('inf')

        # 시각화
        out=warped.copy()
        cv2.drawContours(out,[cnt],-1,(0,255,0),2)
        st.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),use_column_width=True)

        st.success(f"✅ 인식: **{snack}**")
        st.markdown(f"- 비율: `{ratio:.2f}`  원형도: `{roundness:.2f}`  Hue: `{hue}`")
        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)
        st.markdown("#### ⚠️ 하루 권장 최대 섭취 개수")
        st.write(f"- 당 기준: 최대 **{int(max_su)}개**")
        st.write(f"- 나트륨 기준: 최대 **{int(max_na)}개**")

    except Exception as e:
        st.error(f"분석 실패: {e}")
