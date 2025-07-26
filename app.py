# app.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# 1) 클래스 레이블
LABELS = [
    "뻥튀기",
    "데미소다",
    "쫀디기",
    "메가톤",
    "월드콘",
    "조리퐁",
    "미쯔블랙",
    "앙크림빵"
]

# 2) 영양·알레르기 정보
nutrition_db = {
    "뻥튀기":    {"열량":100, "당":1,  "나트륨":50,  "알레르기":"없음"},
    "데미소다":  {"열량":140, "당":35, "나트륨":10,  "알레르기":"없음"},
    "쫀디기":    {"열량":150, "당":20, "나트륨":80,  "알레르기":"밀"},
    "메가톤":    {"열량":220, "당":18, "나트륨":150, "알레르기":"우유, 대두, 밀"},
    "월드콘":    {"열량":200, "당":22, "나트륨":120, "알레르기":"우유, 밀"},
    "조리퐁":    {"열량":140, "당":10, "나트륨":100, "알레르기":"밀"},
    "미쯔블랙":  {"열량":180, "당":12, "나트륨":90,  "알레르기":"우유, 밀"},
    "앙크림빵":  {"열량":250, "당":15, "나트륨":180, "알레르기":"우유, 계란, 밀, 대두"}
}

# 3) 모델 로드
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("snack_classifier.h5")

model = load_model()

# 4) Streamlit UI
st.set_page_config(page_title="푸드스캐너 (딥러닝)", layout="centered")
st.title("📷 푸드스캐너 (딥러닝 버전)")
st.caption("사진을 올리면 8개 스낵 중 하나로 분류하고, 영양·알레르기 정보를 보여줍니다.")

uploaded = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 전처리
    x = np.array(img.resize((224,224))) / 255.0
    x = x.astype(np.float32)[np.newaxis, ...]

    # 예측
    preds = model.predict(x)[0]
    idx   = int(np.argmax(preds))
    score = float(preds[idx] * 100)

    snack = LABELS[idx]
    info  = nutrition_db[snack]

    st.success(f"✅ 인식 결과: **{snack}** ({score:.1f}%)")
    st.markdown("#### ℹ️ 영양·알레르기 정보")
    st.table(info)
