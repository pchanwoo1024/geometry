import streamlit as st
from ultralytics import YOLO
import cv2, numpy as np
import pandas as pd

# ─── 1) 클래스명 & 영양·알레르기 데이터 ─────────────────────────
CLASSES = [
    "뻥튀기","데미소다","쫀디기","메가톤",
    "월드콘","조리퐁","미쯔블랙","앙크림빵"
]
NUTRITION = {
    "뻥튀기":   {"열량(kcal)":100,"탄수(g)":24,"단백질(g)":2,"지방(g)":0.5,"당(g)":1,  "나트륨(mg)":50,  "알레르기":"없음"},
    "데미소다": {"열량(kcal)":140,"탄수(g)":35,"단백질(g)":0,"지방(g)":0,  "당(g)":35, "나트륨(mg)":10,  "알레르기":"없음"},
    "쫀디기":   {"열량(kcal)":150,"탄수(g)":30,"단백질(g)":3,"지방(g)":1,  "당(g)":20, "나트륨(mg)":80,  "알레르기":"밀"},
    "메가톤":   {"열량(kcal)":220,"탄수(g)":27,"단백질(g)":2.5,"지방(g)":12,"당(g)":18, "나트륨(mg)":150,"알레르기":"우유,대두,밀"},
    "월드콘":   {"열량(kcal)":200,"탄수(g)":25,"단백질(g)":3,"지방(g)":10, "당(g)":22, "나트륨(mg)":120,"알레르기":"우유,밀"},
    "조리퐁":   {"열량(kcal)":140,"탄수(g)":22,"단백질(g)":3,"지방(g)":2,  "당(g)":10, "나트륨(mg)":100,"알레르기":"밀"},
    "미쯔블랙": {"열량(kcal)":180,"탄수(g)":20,"단백질(g)":2,"지방(g)":8,  "당(g)":12, "나트륨(mg)":90, "알레르기":"우유,밀"},
    "앙크림빵": {"열량(kcal)":250,"탄수(g)":30,"단백질(g)":5,"지방(g)":8,  "당(g)":15, "나트륨(mg)":180,"알레르기":"우유,계란,밀,대두"},
}
SUGAR_MAX   = 50    # g
SODIUM_MAX  = 2000  # mg

# ─── 2) 모델 로드 ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    # best.pt: 여러분이 train/val 한 YOLOv8 detection 모델
    return YOLO("best.pt")

model = load_model()

# ─── 3) Streamlit UI ────────────────────────────────────────────
st.set_page_config(page_title="딥푸드스캐너", layout="centered")
st.title("📷 딥푸드스캐너 (YOLOv8 Detection)")
st.caption("딥러닝으로 8개 스낵을 바로 검출 → 분류 → 영양·알레르기·권장량 안내")

uploaded = st.file_uploader("간식 사진 업로드", type=["jpg","png","jpeg"])
if uploaded:
    # 이미지 읽기
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # 모델 추론
    results = model(img)[0]

    # 결과가 없으면
    if results.boxes.shape[0] == 0:
        st.error("간식을 검출하지 못했습니다.")
    else:
        # 신뢰도 가장 높은 박스 하나만
        scores = results.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(scores))
        box = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)
        cls = int(results.boxes.cls[best_idx].cpu().numpy())
        name = CLASSES[cls]

        # 잘린 스낵
        x1,y1,x2,y2 = box
        crop = img[y1:y2, x1:x2]

        # 화면에 박스 & 레이블 출력
        disp = img.copy()
        cv2.rectangle(disp, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(disp, name, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        st.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB),
                 use_container_width=True)

        # 영양 정보 테이블
        st.markdown(f"## ✅ {name} 인식 완료!")
        df = pd.DataFrame.from_dict(NUTRITION[name], orient="index", columns=["값"])
        st.table(df)

        # 권장 섭취 개수
        sugar = NUTRITION[name]["당(g)"]
        sodium= NUTRITION[name]["나트륨(mg)"]
        max_s = SUGAR_MAX  // sugar   if sugar>0   else "∞"
        max_n = SODIUM_MAX// sodium  if sodium>0  else "∞"
        st.markdown("#### ⚠️ 하루 최대 권장 섭취 개수")
        st.write(f"- 당 기준: **{max_s}개**  |  나트륨 기준: **{max_n}개**")
