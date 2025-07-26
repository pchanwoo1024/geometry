# app.py

import streamlit as st
import cv2
import numpy as np
from snack_utils import analyze_snack_image

st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너 (YOLO + 기하분류)")
st.caption("YOLO로 객체검출 후 라벨을 펼쳐서 8개 스낵을 분류·영양안내까지!")

uploaded = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    try:
        snack, ratio, roundness, hue, taper, info, ms, mn, out = analyze_snack_image(img)

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 비율: `{ratio:.2f}`  원형도: `{roundness:.2f}`")
        st.markdown(f"- Hue: `{hue}`  Taper: `{taper:.2f}`")

        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)

        st.markdown("#### ⚠️ 하루 권장 최대 섭취 개수")
        st.write(f"- 당 기준: **{ms}개**")
        st.write(f"- 나트륨 기준: **{mn}개**")

    except Exception as e:
        st.error(f"분석 실패: {e}")
