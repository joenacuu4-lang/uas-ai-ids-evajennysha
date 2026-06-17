
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="AI IDS - Eva Jennysha", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS)")
st.markdown("**Peneliti:** Eva Jennysha | **Kelas:** Informatika 4C")
st.markdown("Metode: **Lightweight CNN dengan Squeeze-and-Excitation Attention**")

@st.cache_resource
def load_resources():
    # Path relatif untuk Streamlit Cloud / lokal
    paths = ["ids_model.h5", "06_Model/ids_model.h5"]
    for p in paths:
        if os.path.exists(p):
            return tf.keras.models.load_model(p), joblib.load(p.replace("ids_model.h5","scaler.pkl")), joblib.load(p.replace("ids_model.h5","le_dict.pkl"))
    st.error("Model tidak ditemukan!")
    return None, None, None

model, scaler, le_dict = load_resources()

if model:
    st.sidebar.header("Input Parameter Jaringan")
    duration = st.sidebar.number_input("Duration", value=0.0)
    src_bytes = st.sidebar.number_input("Source Bytes", value=0.0)
    dst_bytes = st.sidebar.number_input("Destination Bytes", value=0.0)

    if st.button("Deteksi Sekarang"):
        input_list = [duration, 'tcp', 'http', 'SF', src_bytes, dst_bytes] + [0]*35
        input_df = pd.DataFrame([input_list[:41]])

        for col_idx, col_name in enumerate(input_df.columns):
            if col_name in le_dict:
                try:
                    input_df.iloc[0, col_idx] = le_dict[col_name].transform([str(input_df.iloc[0, col_idx])])[0]
                except:
                    input_df.iloc[0, col_idx] = 0
            else:
                input_df.iloc[0, col_idx] = pd.to_numeric(input_df.iloc[0, col_idx], errors='coerce')

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0][0]

        st.divider()
        if prediction > 0.5:
            st.error(f"⚠️ SERANGAN TERDETEKSI! ({prediction*100:.2f}%)")
        else:
            st.success(f"✅ TRAFIK NORMAL ({(1-prediction)*100:.2f}%)")
