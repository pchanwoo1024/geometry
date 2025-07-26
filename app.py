# app.py

import streamlit as st
import os
import cv2
import numpy as np
from snack_utils import analyze_snack_image

# --- 디버깅: 파일 목록 찍어보기 ---
st.write("현재 작업 디렉터리:", os.getcwd())
st.write("디렉터리 내 파일들:", os.listdir())

# --- 앱 설정 ---
st.set_page_config(page_title="푸드스캐너", layout="centered")
st.title("📷 푸드스캐너 (YOLO‑Seg + 기하분류)")
st.caption("분석 시작 전에 위에 파일 목록이 보이는지 확인하세요.")

uploaded = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    try:
        snack, ratio, roundness, hue, taper, info, max_sug, max_sod, out = analyze_snack_image(img)

        st.image(
            cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )
        st.success(f"✅ 인식된 간식: **{snack}**")
        st.markdown(f"- 비율: `{ratio:.2f}`  원형도: `{roundness:.2f}`")
        st.markdown(f"- Hue: `{hue}`  Taper: `{taper:.2f}`")
        st.markdown("#### ℹ️ 영양·알레르기 정보")
        st.table(info)
        st.markdown("#### ⚠️ 하루 최대 권장 섭취 개수")
        st.write(f"- 당 기준: **{max_sug}개**")
        st.write(f"- 나트륨 기준: **{max_sod}개**")

    except Exception as e:
        st.error("분석에 실패했습니다. 아래에서 상세 오류를 확인하세요.")
        st.exception(e)
