import streamlit as st
import cv2
import numpy as np
from snack_utils import analyze_snack_image
from food_data import nutrition_allergy_db

st.set_page_config(page_title="스낵 분류기", layout="centered")
st.title("📷 간식 이미지 인식기")
st.caption("사진을 업로드하면 어떤 간식인지 알려주고, 영양 정보도 보여드려요!")

uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    try:
        snack, ratio, roundness, info, out_img = analyze_snack_image(image)
        st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), caption="보정된 간식 이미지", use_column_width=True)
        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 비율: `{ratio:.2f}`\n- 원형도: `{roundness:.2f}`")
        st.markdown("#### ℹ️ 영양 및 알레르기 정보")
        st.table(info)
    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")