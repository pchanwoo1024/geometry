# app.py

import streamlit as st
import cv2
import numpy as np
from snack_utils import analyze_snack_image

st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너 (YOLO‑Seg + 기하분류)")
st.caption("YOLOv8 Segmentation으로 캔을 분할 → 라벨 정사영 → 8개 스낵 분류·영양 안내까지!")

uploaded = st.file_uploader("스낵 사진 업로드", type=["jpg", "png", "jpeg"])
if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    try:
        # 분석: snack, ratio, roundness, hue, taper, info, max_sugar, max_sodium, warped_out
        snack, ratio, roundness, hue, taper, info, max_sug, max_sod, out = analyze_snack_image(img)

        # 결과 이미지
        st.image(
            cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        # 텍스트 정보
        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 비율: `{ratio:.2f}`  원형도: `{roundness:.2f}`")
        st.markdown(f"- Hue: `{hue}`  Taper: `{taper:.2f}`")
        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)
        st.markdown("#### ⚠️ 하루 권장 최대 섭취 개수")
        st.write(f"- 당 기준: **{max_sug}개**")
        st.write(f"- 나트륨 기준: **{max_sod}개**")

    except Exception as e:
        st.error("분석에 실패했습니다. 아래에서 상세 오류를 확인하세요.")
        st.exception(e)
