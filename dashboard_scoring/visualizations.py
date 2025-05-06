import streamlit as st
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import lime
import lime.lime_tabular
import traceback
from PIL import Image
import io
from accessibility import get_accessible_color_for_score, apply_accessible_style, add_alt_text



SEUIL_METIER = 0.19

# Affichage de la jauge de score
def plot_score_gauge(score_data, threshold=SEUIL_METIER):
    score = score_data["score"]
    color, status = get_accessible_color_for_score(score, threshold)

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score,
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold], 'color': "#66bb6a"},
                {'range': [threshold, 1], 'color': "#ef5350"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.add_annotation(
        x=0.5, y=0.25,
        text=f"<b style='font-size:25px'>{status}</b><br><span style='font-size:14px'>Seuil d'acceptation = {int(threshold * 100)}%</span>",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="center"
    )

    fig = apply_accessible_style(fig)
    add_alt_text("Jauge de score indiquant la probabilité de défaut du client avec code couleur.")
    st.plotly_chart(fig, use_container_width=True)
    return status


# Feature importance locale avec LIME
def plot_lime_local(pipeline, client_data, all_clients_data, expected_score):
    try:
        if client_data is None or client_data.empty:
            raise ValueError("Les données client sont vides ou manquantes")
        if all_clients_data is None or all_clients_data.empty:
            raise ValueError("Les données de tous les clients sont vides ou manquantes")
        if pipeline is None:
            raise ValueError("Le modèle est manquant")

        # Vérifier si le pipeline a des feature_names_in_ ou si nous devons utiliser les colonnes du client_data
        if hasattr(pipeline, 'feature_names_in_'):
            feature_cols = list(pipeline.feature_names_in_)
        elif hasattr(pipeline, 'feature_names'):
            feature_cols = list(pipeline.feature_names)
        else:
            feature_cols = client_data.columns.tolist()
            # Si SK_ID_CURR est dans les colonnes, on l'exclut
            if 'SK_ID_CURR' in feature_cols:
                feature_cols.remove('SK_ID_CURR')

        # Vérification que toutes les colonnes nécessaires sont disponibles
        missing_cols = [col for col in feature_cols if col not in all_clients_data.columns]
        if missing_cols:
            available_cols = [col for col in feature_cols if col in all_clients_data.columns]
            st.warning(f"Colonnes manquantes dans all_clients_data (utilisant {len(available_cols)}/{len(feature_cols)} colonnes): {missing_cols}")
            # Utiliser uniquement les colonnes disponibles
            feature_cols = available_cols

        client_id = None
        if "SK_ID_CURR" in all_clients_data.columns and client_data.index.size > 0:
            if "SK_ID_CURR" in client_data.columns:
                client_id = client_data["SK_ID_CURR"].iloc[0]
            else:
                client_idx = client_data.index[0]
                if client_idx in all_clients_data.index:
                    client_id = all_clients_data.loc[client_idx, "SK_ID_CURR"]

        X = all_clients_data[feature_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)

        if X.isna().any().any():
            X.fillna(0, inplace=True)

        client_idx = None
        if client_id is not None:
            matching_rows = all_clients_data[all_clients_data["SK_ID_CURR"] == int(client_id)]
            if not matching_rows.empty:
                client_idx = matching_rows.index[0]

        if client_idx is None:
            client_idx = client_data.index[0]
            if client_idx not in all_clients_data.index:
                client_idx = X.index[0]

        if client_idx not in X.index:
            raise ValueError(f"L'index client {client_idx} n'existe pas dans le dataset nettoyé")

        client_values = X.loc[client_idx].values
        if np.isnan(client_values).any() or np.isinf(client_values).any():
            client_values = np.nan_to_num(client_values)

        # Création de l'explainer LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=feature_cols,
            class_names=["Non défaut", "Défaut"],
            verbose=True,
            mode='classification'
        )

        # Fonction de prédiction adaptée au modèle chargé
        def predict_fn(x):
            # Vérifier si le modèle possède predict_proba ou s'il faut utiliser predict
            if hasattr(pipeline, 'predict_proba'):
                return pipeline.predict_proba(pd.DataFrame(x, columns=feature_cols))
            else:
                # Fallback vers une simple prédiction binaire si predict_proba n'existe pas
                # Prédictions binaires converties en pseudo-probabilités [1-p, p]
                preds = pipeline.predict(pd.DataFrame(x, columns=feature_cols))
                return np.column_stack((1-preds, preds))

        # Génération de l'explication
        explanation = explainer.explain_instance(
            client_values,
            predict_fn,
            num_features=8
        )

        # Création du graphique
        fig = explanation.as_pyplot_figure()
        plt.title(f'Facteurs influençant la prédiction)', fontsize=13)
        plt.tight_layout()
        fig = apply_accessible_style(fig)
        add_alt_text("Graphique expliquant localement la prédiction du modèle pour ce client avec LIME.")

        return fig

    except Exception as e:
        st.error(f"Erreur détaillée dans LIME: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Erreur lors de l'analyse LIME:\n{str(e)}",
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title("Erreur d'interprétation locale", fontsize=14, color='red')
        add_alt_text("Graphique expliquant localement la prédiction du modèle pour ce client avec LIME.")

        return fig

def plot_feature_distribution(df, feature, client_value):
    fig = plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=feature, kde=True, stat="density", element="step", fill=True)
    plt.axvline(client_value, color='black', linestyle='--', label='Client')
    plt.title(f"Distribution de {feature} (tous clients)")
    plt.legend()
    fig = apply_accessible_style(fig)
    add_alt_text(f"Histogramme de la distribution de la variable {feature} pour tous les clients, avec une ligne indiquant la valeur du client.")

    return fig

# Salaire Median par target avec positionnement client
def plot_variable_by_target(df, variable="AMT_INCOME_TOTAL", client_value=None):
    import plotly.graph_objects as go
    import plotly.express as px

    df = df.copy()
    df["AGE"] = (-df["DAYS_BIRTH"] // 365).astype(int)
    bins = [20, 30, 40, 50, 60, 70, 80]
    labels = [f"[{a}, {b})" for a, b in zip(bins[:-1], bins[1:])]
    df["AGE_BIN"] = pd.cut(df["AGE"], bins=bins, labels=labels, right=False, include_lowest=True)

    if variable == "AMT_INCOME_TOTAL":
        medianes = df.groupby("TARGET")[variable].median()
        labels_bar = ["Crédit remboursé", "Défaut de crédit"]
        values = [medianes[0], medianes[1]]
        colors = ["#2ecc71", "#e74c3c"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels_bar,
            y=values,
            marker_color=colors,
            text=[f"{int(val):,}".replace(",", " ") + " €" for val in values],
            textposition="auto",
            showlegend=False
        ))

        if client_value is not None:
            fig.add_trace(go.Scatter(
                x=labels_bar,
                y=[client_value, client_value],
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="Client"
            ))

        fig.update_layout(
            title="Revenu médian selon la cible (TARGET)",
            yaxis_title="Revenu médian (€)",
            xaxis_title="Statut de remboursement",
            height=600
        )
        fig = apply_accessible_style(fig)
        add_alt_text("Barres du revenu médian selon le statut de remboursement, avec ligne de positionnement du client.")

        return fig

    elif variable == "AGE":
        # Afficher le taux de défaut pour chaque tranche
        taux_par_tranche = df.groupby("AGE_BIN")["TARGET"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=taux_par_tranche["AGE_BIN"].astype(str),
            y=taux_par_tranche["TARGET"],
            marker_color="#e74c3c",
            text=[f"{p:.1%}" for p in taux_par_tranche["TARGET"]],
            textposition="auto",
            showlegend=False
        ))

        if client_value is not None:
            client_age = int(client_value)
            client_bin = pd.cut([client_age], bins=bins, labels=labels, right=False)[0]

            # Ajout d’une ligne verticale pour indiquer le client
            fig.add_trace(go.Scatter(
                x=[str(client_bin)],
                y=[taux_par_tranche[taux_par_tranche["AGE_BIN"] == client_bin]["TARGET"].values[0]],
                mode="markers",
                marker=dict(size=12, color="black", symbol="x"),
                name="Client"
            ))

        fig.update_layout(
            title="Taux de défaut par tranche d’âge",
            yaxis_title="Taux de défaut (%)",
            xaxis_title="Tranches d’âge",
            height=600
        )
        fig = apply_accessible_style(fig)
        add_alt_text("Taux de défaut de crédit par tranche d'âge, avec indication de la tranche du client.")

        return fig

    else:
        raise ValueError("Variable non prise en charge")

# Analyse bivariée Revenus / annuités
def plot_bivariate_analysis_with_density(df, feature_x, feature_y, target_col, client_row):
    import plotly.express as px
    import plotly.graph_objects as go

    # Valeurs du client
    client_x = float(client_row[feature_x])
    client_y = float(client_row[feature_y])

    # Convertir TARGET en string pour affichage
    df[target_col] = df[target_col].astype(str)

    # Supprimer les valeurs extrêmes (1% - 99%)
    q_low_x, q_high_x = df[feature_x].quantile([0.01, 0.99])
    q_low_y, q_high_y = df[feature_y].quantile([0.01, 0.99])
    df_filtered = df[
        (df[feature_x].between(q_low_x, q_high_x)) &
        (df[feature_y].between(q_low_y, q_high_y))
    ].copy()

    # Nombre de clients affichés
    nb_clients = len(df_filtered)

    # Créer le graphique
    fig = px.scatter(
        df_filtered,
        x=feature_x,
        y=feature_y,
        color=target_col,
        color_discrete_map={"0": "#2ecc71", "1": "#e74c3c"},
        category_orders={target_col: ["0", "1"]},
        opacity=0.5,
        labels={
            feature_x: "Revenus",
            feature_y: "Annuités",
            target_col: "Cible (TARGET)"
        },
        height=600,
        title=f"Analyse bivariée sur les clients filtrés ({nb_clients} affichés)"
    )

    # Ajouter le point du client
    fig.add_trace(go.Scatter(
        x=[client_x],
        y=[client_y],
        mode='markers',
        marker=dict(size=12, color='black', symbol='x'),
        name="Client"
    ))

    fig = apply_accessible_style(fig)
    add_alt_text("Nuage de points Revenus vs Annuités (valeurs extrêmes supprimées) avec le client en noir.")

    return fig


# Analyse bivariée comportement de paiements
def plot_bivariate_payment_behavior(df, feature_x, feature_y, target_col, client_row):
    import plotly.express as px
    import plotly.graph_objects as go

    # Valeurs du client
    client_x = float(client_row[feature_x])
    client_y = float(client_row[feature_y])

    # Convertir TARGET en string pour affichage
    df[target_col] = df[target_col].astype(str)

    # Supprimer les valeurs extrêmes (1% - 99%)
    q_low_x, q_high_x = df[feature_x].quantile([0.01, 0.99])
    q_low_y, q_high_y = df[feature_y].quantile([0.01, 0.99])
    df_filtered = df[
        (df[feature_x].between(q_low_x, q_high_x)) &
        (df[feature_y].between(q_low_y, q_high_y))
    ].copy()

    # Nombre de clients retenus
    nb_clients = len(df_filtered)

    # Créer le graphique
    fig = px.scatter(
        df_filtered,
        x=feature_x,
        y=feature_y,
        color=target_col,
        color_discrete_map={"0": "#2ecc71", "1": "#e74c3c"},
        category_orders={target_col: ["0", "1"]},
        opacity=0.3,
        labels={
            feature_x: "Moy. proportion payé / dû",
            feature_y: "Moy. mensualités",
            target_col: "Cible (TARGET)"
        },
        height=600,
        title=f"Analyse bivariée sur les clients filtrés ({nb_clients} affichés)"
    )

    # Ajouter le point du client
    fig.add_trace(go.Scatter(
        x=[client_x],
        y=[client_y],
        mode='markers',
        marker=dict(size=12, color='black', symbol='x'),
        name="Client"
    ))

    fig = apply_accessible_style(fig)
    add_alt_text("Nuage de points comportement de paiement : proportion payé vs mensualités moyennes, avec le point du client (valeurs extrêmes supprimées).")

    return fig
