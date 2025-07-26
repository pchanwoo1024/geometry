import streamlit as st
import cv2
import numpy as np
from snack_utils import analyze_snack_image

st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너 (YOLO + 기하분류)")
st.caption("YOLO 객체검출 → 라벨 정사영 → 8개 스낵 분류·영양 안내까지!")

uploaded = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if uploaded:
    img_b = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img   = cv2.imdecode(img_b, cv2.IMREAD_COLOR)

    try:
        snack, r, rd, hue, taper, info, ms, mn, out = analyze_snack_image(img)

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 비율: `{r:.2f}`  원형도: `{rd:.2f}`")
        st.markdown(f"- Hue: `{hue}`  Taper: `{taper:.2f}`")
        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)
        st.markdown("#### ⚠️ 하루 최대 권장 섭취 개수")
        st.write(f"- 당 기준: **{ms}개**")
        st.write(f"- 나트륨 기준: **{mn}개**")

    except Exception as e:
        st.error("분석 실패, 상세 정보를 확인하려면 아래를 펼쳐보세요.")
        st.exception(e)
