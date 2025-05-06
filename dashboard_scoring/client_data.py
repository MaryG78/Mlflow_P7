import pandas as pd
import numpy as np
import streamlit as st
import os
import requests
import joblib


# Fonction utilitaire pour trouver le bon chemin des données
def find_data_path(filename):
    # Liste des chemins possibles à vérifier
    possible_paths = [
        # Chemin relatif standard
        os.path.join(os.path.dirname(__file__), "saved_data", filename),
        # Chemin relatif depuis la racine du projet (pour Streamlit Cloud)
        os.path.join("saved_data", filename),
        # Chemin absolu (pour les tests locaux)
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "saved_data", filename)
    ]
    
    # Vérifier chaque chemin et renvoyer le premier qui existe
    for path in possible_paths:
        if os.path.exists(path):
            return path
        else:
            st.write(f"Chemin testé mais non trouvé: {path}")
    
    # Si aucun chemin n'est trouvé, renvoyer le premier (qui déclenchera une erreur plus informative)
    st.error(f"Fichier non trouvé: {filename}. Vérifiez que le dossier 'saved_data' existe et contient ce fichier.")
    return possible_paths[0]

# Définition des chemins de fichiers
CLIENTS_PATH = find_data_path("Base_client.parquet")
APPLICATION_TRAIN_PATH = find_data_path("application_train.csv")

@st.cache_data
def load_all_clients():
    try:
        df = pd.read_parquet(CLIENTS_PATH)
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement de Base_client.parquet: {e}")
        # Retourner un DataFrame vide pour éviter des erreurs en cascade
        return pd.DataFrame(columns=["SK_ID_CURR"])

def get_client_list():
    df = load_all_clients()
    if df.empty:
        return ["Erreur de chargement des données"]
    return df["SK_ID_CURR"].unique().tolist()

def get_client_data(SK_ID_CURR):
    df = load_all_clients()
    df = df[df["SK_ID_CURR"] == int(SK_ID_CURR)]
    if df.empty:
        return pd.DataFrame()
    return df.drop(columns=["SK_ID_CURR"])

@st.cache_data
def load_raw_applications():
    try:
        df = pd.read_csv(APPLICATION_TRAIN_PATH)
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
        clients_df = load_all_clients()
        ids = clients_df["SK_ID_CURR"].unique()
        return df[df["SK_ID_CURR"].isin(ids)]
    except Exception as e:
        st.error(f"Erreur lors du chargement de application_train.csv: {e}")
        return pd.DataFrame()

def get_raw_client_info(sk_id_curr):
    df = load_raw_applications()
    if df.empty:
        return pd.DataFrame()
    return df[df["SK_ID_CURR"] == int(sk_id_curr)]

@st.cache_data
def load_application_train():
    try:
        df = pd.read_csv(APPLICATION_TRAIN_PATH)
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement de application_train.csv: {e}")
        return pd.DataFrame()

# Fonction pour charger le modèle depuis GitHub
@st.cache_resource
def load_model_from_github():
    url = "https://raw.githubusercontent.com/MaryG78/Mlflow_P7/main/dashboard_scoring/saved_data/best_model_20_features.joblib"
    model_path = "saved_data/best_model_20_features.joblib"
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Erreur lors du téléchargement du modèle: {response.status_code}")
            return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None