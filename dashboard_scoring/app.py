import streamlit as st
import os
import requests
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback
from api_utils import get_client_score
from PIL import Image as PILImage
from visualizations import (
    plot_score_gauge,
    plot_lime_local,
    plot_feature_distribution,
    plot_variable_by_target,
    plot_bivariate_analysis_with_density,
    plot_bivariate_payment_behavior
)
from client_data import get_client_list, get_client_data, get_raw_client_info, load_all_clients, load_application_train, load_model_from_github

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Dashboard Scoring - Relation Client")

# Vérification des chemins de fichiers disponibles
if st.sidebar.checkbox("Vérifier les fichiers disponibles (Debug)"):
    st.sidebar.write("Répertoire courant:", os.getcwd())
    st.sidebar.write("Contenu du répertoire courant:", os.listdir())
    if os.path.exists("saved_data"):
        st.sidebar.write("Contenu du dossier saved_data:", os.listdir("saved_data"))
    else:
        st.sidebar.error("Le dossier saved_data n'existe pas!")

# Liste des clients
client_list = get_client_list()
if not client_list or client_list[0] == "Erreur de chargement des données":
    st.error("Impossible de charger la liste des clients. Vérifiez que le fichier Base_client.parquet existe dans le dossier saved_data.")
    st.stop()

client_id = st.selectbox("Sélectionnez un numéro de client :", client_list)

if client_id:
    try:
        client_data, score_data = get_client_score(client_id)
        if client_data.empty or len(client_data) != 1:
            st.error("Erreur : données client introuvables ou multiples.")
            st.stop()

        raw_client_info = get_raw_client_info(client_id)
        if raw_client_info.empty:
            st.error(f"Données brutes introuvables pour le client {client_id}")
            st.stop()

        raw_client_info = raw_client_info.iloc[0]
        df_all_clients = load_all_clients()

        model = load_model_from_github()

        feature_cols = list(client_data.columns)
        X = df_all_clients[feature_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        df_all_clients = df_all_clients.loc[X.index].copy()

        # Initialiser score avec NaN, puis injecter uniquement le score du client sélectionné
        df_all_clients["score"] = np.nan
        df_all_clients.loc[df_all_clients["SK_ID_CURR"] == int(client_id), "score"] = score_data["score"]


    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        st.stop()

    # Affichage en 2 colonnes
    col1, col2 = st.columns([1, 1])

    # Colonne 'Informations client'
    with col1:
        st.markdown("### 🧾 Informations client")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown(f"**Âge :** {int(-raw_client_info['DAYS_BIRTH'] // 365)} ans")
        st.markdown(f"**Type de revenu :** {raw_client_info['NAME_INCOME_TYPE']}")
        st.markdown(f"**Statut familial :** {raw_client_info['NAME_FAMILY_STATUS']}")
        st.markdown(f"**Nombre de personnes dans le foyer :** {raw_client_info['CNT_FAM_MEMBERS']}")
        st.markdown("---")
        st.markdown(f"**Montant total des revenus :** {raw_client_info['AMT_INCOME_TOTAL']:,.0f} €")
        st.markdown(f"**Possède une voiture :** {'Oui' if raw_client_info['FLAG_OWN_CAR'] == 'Y' else 'Non'}")
        st.markdown(f"**Possède un bien immobilier :** {'Oui' if raw_client_info['FLAG_OWN_REALTY'] == 'Y' else 'Non'}")
    
    # Colonne 'Score'
    with col2:
        st.markdown("<div style='font-size:22px; font-weight:600; margin-bottom: -23px'>📈 Résultat du scoring</div>""", unsafe_allow_html=True)

        plot_score_gauge(score_data)
        score = score_data["score"]
        st.markdown(f"<div style='text-align:center; margin-top: -50px; font-size:20px'><strong>Probabilité de défaut de paiement :</strong> {score:.1%}</div>""", unsafe_allow_html=True)

    #Interpretation du score (feature importance)
    col_globale, col_locale = st.columns([1, 1])

    # Interprétation globale
    with col_globale:
        st.subheader("🌍 Importance globale des données")
        with st.expander("Voir l'interprétation globale"):
            # Liste des chemins possibles
            possible_image_paths = [
                "dashboard_scoring/assets/global_importance.png",
                "assets/global_importance.png",
                "global_importance.png",
                "saved_data/global_importance.png"
            ]
            
            image_found = False
            for image_path in possible_image_paths:
                try:
                    if os.path.exists(image_path):
                        image = PILImage.open(image_path)
                        st.image(image, caption="Importance globale des variables (SHAP)", use_container_width=True)
                        image_found = True
                        break
                except Exception as e:
                    pass
            
            if not image_found:
                st.error("Image d'importance globale non trouvée. Chemins vérifiés : " + ", ".join(possible_image_paths))

    # Interprétation niveau client
    with col_locale:
        st.subheader("📊 Interprétation locale du score")
        with st.expander("Voir l’interprétation locale"):
            try:
                # Vérifier que le modèle est bien chargé
                if model is None:
                    st.warning("Le modèle n'a pas pu être chargé. L’interprétation locale n’est pas disponible.")
                    st.stop()

                # Générer l’explication LIME
                fig_lime = plot_lime_local(
                    pipeline=model,
                    client_data=client_data,
                    all_clients_data=df_all_clients,
                    expected_score=score_data["score"]
                )
                st.pyplot(fig_lime)

            except Exception as e:
                st.error(f"Erreur LIME : {e}")
                import traceback
                st.code(traceback.format_exc())


    st.subheader("📊 Positionnement du client par rapport aux revenus et à l'âge")
    try:
        app_train = load_application_train()
        variable_choice = st.selectbox("Choisissez la donnée à afficher :", options=["Revenus", "Âge"])
        if variable_choice == "Revenus":
            selected_var = "AMT_INCOME_TOTAL"
            client_value = raw_client_info["AMT_INCOME_TOTAL"]
        else:
            selected_var = "AGE"
            client_value = int(-raw_client_info["DAYS_BIRTH"] // 365)

        fig_var = plot_variable_by_target(app_train, selected_var, client_value)
        st.plotly_chart(fig_var, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du graphique: {e}")

    st.subheader("📈 Statut du crédit en fonction des revenus et de l'annuité")
    try:
        fig_biv = plot_bivariate_analysis_with_density(
            df=app_train,
            feature_x="AMT_INCOME_TOTAL",
            feature_y="AMT_ANNUITY",
            target_col="TARGET",
            client_row=raw_client_info
        )
        st.plotly_chart(fig_biv, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'analyse bivariée: {e}")

    st.subheader("📉 Statut du crédit en fonction des mensualités et des retards")
    try:
        df_clients = load_all_clients()
        df_target = load_application_train()[["SK_ID_CURR", "TARGET"]]
        df_merged = pd.merge(df_clients, df_target, on="SK_ID_CURR", how="left")

        client_row = df_merged[df_merged["SK_ID_CURR"] == int(client_id)]

        fig_payment = plot_bivariate_payment_behavior(
            df=df_merged,
            feature_x="INSTAL_PAYMENT_PERC_MEAN",
            feature_y="APPROVED_AMT_ANNUITY_MEAN",
            target_col="TARGET",
            client_row=client_row
        )
        st.plotly_chart(fig_payment, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du graphique de comportement de paiement : {e}")

    st.subheader("✏️ Simulation : modification des caractéristiques client")
    try:
        FEATURE_LABELS = {
            "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": "Moy. retraits DAB carte crédit",
            "BURO_DAYS_CREDIT_MIN": "Ancienneté crédit le plus ancien",
            "INSTAL_AMT_PAYMENT_MIN": "Paiement minimum effectué",
            "APPROVED_DAYS_DECISION_MIN": "Décision la plus ancienne (crédits approuvés)",
            "INSTAL_DPD_MEAN": "Moy. jours de retard paiement",
            "EXT_SOURCE_3_DAYS_BIRTH": "Source externe 3 x Âge",
            "APPROVED_AMT_ANNUITY_MEAN": "Moy. mensualités (crédits approuvés)",
            "INSTAL_DAYS_ENTRY_PAYMENT_MAX": "Max. décalage date d'enregistrement paiement",
            "CC_CNT_DRAWINGS_ATM_CURRENT_VAR": "Var. retraits DAB carte crédit",
            "EXT_SOURCE_1_EXT_SOURCE_3": "Source externe 1 x 3",
            "EXT_SOURCE_2_EXT_SOURCE_3": "Source externe 2 x 3",
            "CREDIT_TERM": "Durée du crédit",
            "INSTAL_DAYS_ENTRY_PAYMENT_SUM": "Somme décalage dates d'enregistrement paiements",
            "INSTAL_PAYMENT_DIFF_MEAN": "Moy. des écarts montant dû – payé",
            "INSTAL_AMT_PAYMENT_MAX": "Paiement maximum effectué",
            "BURO_DAYS_CREDIT_ENDDATE_MEAN": "Moy. jours jusqu'à fin des crédits",
            "INSTAL_PAYMENT_PERC_MEAN": "Moy. proportion payé / dû",
            "INSTAL_AMT_INSTALMENT_MAX": "Montant maximum dû",
            "EXT_SOURCE_3_2": "Source externe 3²",
            "EXT_SOURCE_2_DAYS_BIRTH": "Source externe 2 x Âge",
        }

        df_clients = load_all_clients()
        client_data = df_clients[df_clients["SK_ID_CURR"] == int(client_id)].copy()

        if client_data.empty:
            st.warning("Client introuvable dans Base_client.parquet.")
            st.stop()

        features = client_data.drop(columns=["SK_ID_CURR"])
        new_values = {}

        for col in features.columns:
            label = FEATURE_LABELS.get(col, col)
            val = features[col].values[0]
            if np.issubdtype(features[col].dtype, np.number):
                new_values[col] = st.number_input(label, value=float(val))
            else:
                new_values[col] = st.text_input(label, value=str(val))

        simulated_df = pd.DataFrame([new_values])
        _, simulated_score_data = get_client_score(sk_id_curr=None, custom=True, custom_data=simulated_df)

        st.markdown(
            f"<div style='font-size:18px; color:#000000;'>Score prédit avec les valeurs modifiées : "
            f"<strong>{simulated_score_data['score']:.2%}</strong></div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Erreur lors de la simulation : {e}")
else:
    st.warning("Veuillez sélectionner un client pour afficher le tableau de bord.")