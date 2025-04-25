import streamlit as st
import pandas as pd
import requests

st.title("Prédiction de Score de Crédit")
st.write("Entrez les données du client :")

# Valeurs par défaut
input_row = {
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.5714285969734192,
    "BURO_DAYS_CREDIT_MIN": -2663.0,
    "INSTAL_AMT_PAYMENT_MIN": 8693.639648,
    "APPROVED_DAYS_DECISION_MIN": -428.0,
    "INSTAL_DPD_MEAN": 0.0,
    "EXT_SOURCE_3_DAYS_BIRTH": -5775.94043,
    "APPROVED_AMT_ANNUITY_MEAN": 6589.064941,
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX": -135.0,
    "CC_CNT_DRAWINGS_ATM_CURRENT_VAR": 3.4945051670074463,
    "EXT_SOURCE_1_EXT_SOURCE_3": 0.216394,
    "EXT_SOURCE_2_EXT_SOURCE_3": 0.254168,
    "CREDIT_TERM": 0.039825,
    "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -2704.0,
    "INSTAL_PAYMENT_DIFF_MEAN": 0.0,
    "INSTAL_AMT_PAYMENT_MAX": 8702.099609,
    "BURO_DAYS_CREDIT_ENDDATE_MEAN": -411.666656,
    "INSTAL_PAYMENT_PERC_MEAN": 1.0,
    "INSTAL_AMT_INSTALMENT_MAX": 8702.099609,
    "EXT_SOURCE_3_2": 0.182891,
    "EXT_SOURCE_2_DAYS_BIRTH": -8026.977051
        }
# Affichage d'un champ pour chaque variable
user_input = {}
for key, default_value in input_row.items():
    user_input[key] = st.number_input(key, value=float(default_value))

# Bouton d'envoi
if st.button("Obtenir le score du client"):
    df = pd.DataFrame([user_input]).astype("float32")
    url = "https://guim78.pythonanywhere.com/predict"
    
    response = requests.post(url, json={"dataframe_split": df.to_dict(orient="split")})
    
    if response.status_code == 200:
        prediction = response.json()["predictions"][0]
        st.success(f"Probabilité de remboursement : {prediction:.2f}")
    else:
        st.error(f"Erreur API : {response.status_code}")
        st.write(response.text)