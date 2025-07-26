import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

LABELS = ["뻥튀기","데미소다","쫀디기","메가톤","월드콘","조리퐁","미쯔블랙","앙크림빵"]
NUTRI = { … }  # 기존 nutrition_db

@st.cache_resource
def load_interpreter():
    interp = tflite.Interpreter("snack_classifier.tflite")
    interp.allocate_tensors()
    return interp

interpreter = load_interpreter()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("푸드스캐너 (TFLite)")
up = st.file_uploader("스낵 사진 업로드", type=["jpg","png","jpeg"])
if up:
    img = Image.open(up).convert("RGB")
    st.image(img, use_column_width=True)
    arr = np.array(img.resize((224,224)), dtype=np.float32)[np.newaxis]/255.0

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))
    snack = LABELS[idx]

    st.success(f"✅ 인식: {snack} ({preds[idx]*100:.1f}%)")
    st.table(NUTRI[snack])
