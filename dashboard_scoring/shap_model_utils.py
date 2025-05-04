import os
import joblib
import requests
import streamlit as st

@st.cache_resource
def load_shap_model():
    url = "https://raw.githubusercontent.com/MaryG78/Mlflow_P7/main/model_shap_ready.joblib"
    local_path = "saved_data/model_shap_ready.joblib"

    if not os.path.exists(local_path):
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)

    return joblib.load(local_path)  # Renvoie (model_lgbm, preprocessor)
