import streamlit as st
import cv2
import numpy as np
from snack_utils_improved import analyze_snack_image

st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너")
st.caption("사진을 업로드하면 스낵을 인식하고, 영양·알레르기·안전 섭취 개수까지 알려줘요")

uploaded = st.file_uploader("이미지를 올려주세요", type=["jpg","png","jpeg"])
if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    try:
        res = analyze_snack_image(img)
        out = res["image"]
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.success(f"✅ {res['name']} 으로 인식되었습니다!")
        st.markdown(f"- 가로/세로 비율: `{res['ratio']:.2f}`\n"
                    f"- 원형도: `{res['roundness']:.2f}`\n"
                    f"- 중심점: `{res['centroid']}`\n"
                    f"- 지배 Hue: `{res['hue']}`")

        st.markdown("#### ℹ️ 영양 및 알레르기 정보")
        st.table(res["info"])

        st.markdown("#### ⚠️ 하루 권장 최대 섭취 개수")
        st.write(f"- 당 기준: 최대 **{res['max_sugar']}개**")
        st.write(f"- 나트륨 기준: 최대 **{res['max_sodium']}개**")

    except Exception as e:
        st.error(f"분석 실패: {e}")
