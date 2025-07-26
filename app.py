import streamlit as st
import cv2
import numpy as np
from snack_utils_improved import analyze_snack_image
from food_data import nutrition_allergy_db

st.set_page_config(page_title="스낵 분류기 (개선판)", layout="centered")
st.title("📷 간식 이미지 인식기 (개선된 보정 & 분류)")
st.caption("사진을 업로드하면 기하+색상+중심점을 이용해 정확히 어떤 간식인지 알려드려요!")

uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg","png","jpeg"])
if uploaded:
    # 메모리에서 OpenCV 이미지로 변환
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        # 개선된 분석 함수 호출
        snack, ratio, roundness, (cx,cy), hue, info, out_img = analyze_snack_image(img)

        st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB),
                 caption=f"분류: {snack}  //  Centroid=({cx},{cy})  Hue={hue}",
                 use_column_width=True)

        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 가로/세로 비율: `{ratio:.2f}`  \n"
                    f"- 원형도: `{roundness:.2f}`  \n"
                    f"- 중심점: (`{cx}`, `{cy}`)  \n"
                    f"- 지배 Hue: `{hue}`")

        st.markdown("#### ℹ️ 영양 및 알레르기 정보")
        st.table(info)

    except Exception as e:
        st.error(f"⚠️ 분석 중 오류 발생: {e}")
