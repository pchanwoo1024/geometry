import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

LABELS = ["뻥튀기","데미소다","쫀디기","메가톤","월드콘","조리퐁","미쯔블랙","앙크림빵"]
NUTRI = {
  "뻥튀기": {"열량":100,"당":1,"나트륨":50}, … 
  # 앞서 쓰신 nutrition_db 그대로
}

model = tf.keras.models.load_model("snack_classifier.h5")

st.title("📷 푸드스캐너 (딥러닝 버전)")
up = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if up:
    img = Image.open(up).convert("RGB")
    st.image(img, caption="uploaded", use_column_width=True)
    x = np.array(img.resize((224,224))) / 255.0
    pred = model.predict(x[np.newaxis,...])[0]
    idx = np.argmax(pred)
    snack = LABELS[idx]
    st.success(f"✅ 인식: {snack} ({pred[idx]*100:.1f}%)")
    st.table(NUTRI[snack])
