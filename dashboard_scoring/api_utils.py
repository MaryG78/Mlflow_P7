import numpy as np
import requests
import pandas as pd
import json
from numpy import isnan
from client_data import get_client_data

API_URL = "https://GuiM78.pythonanywhere.com/predict"  

def get_client_score(sk_id_curr, custom=False, custom_data=None):
    if custom and custom_data is not None:
        df = custom_data
    else:
        df = get_client_data(sk_id_curr)

    #  Convertir les NaN en None (compatible JSON)
    df = df.replace({np.nan: None})

    payload = {
        "dataframe_split": df.to_dict(orient="split")
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prediction = response.json()["predictions"][0]
        return df, {"score": prediction}
    else:
        raise ValueError(f"Erreur API : {response.status_code} - {response.text}")