# Projet de Scoring Bancaire - Prédiction du Remboursement de Prêt


## Objectif
Développer un modèle de scoring de crédit afin d’estimer la probabilité de défaut de paiement d’un client, et le déployer via une API accessible sur le cloud.

---

## Contenu du projet

### Modélisation (`P7model_training.ipynb`)
- Préparation des données
- Création de nouvelles variables explicatives (features métier, polynomiales…)
- Test de plusieurs algorithmes (Dummy, RandomForest, XGBoost, LightGBM)
- Gestion du déséquilibre via SMOTE
- Sélection du meilleur modèle via GridSearchCV et score métier
- Optimisation fine avec HyperOpt
- Tracking des expérimentations via **MLflow**
- Export du modèle final au format `.joblib`

### Déploiement de l’API (`api/app.py`)
- Création d’une API Flask capable de recevoir des données et retourner une prédiction
- Chargement du modèle `best_model_20_features.joblib`
- Testée et déployée sur **PythonAnywhere**

### Interface utilisateur (`streamlit_app.py`)
- Interface simple pour entrer les données d’un client
- Affichage de la **probabilité de défaut de paiement**
- Appel de l’API Flask en arrière-plan

### Tests unitaires (`tests/test_api.py`)
- Vérification du format de la réponse de l’API
- Gestion des erreurs (colonnes manquantes, mauvais format…)
- Tests automatisés avec `pytest`

### Monitoring (optionnel)
- Analyse de **data drift** entre `train` et `test` avec **Evidently**
- Rapport HTML intégré

---

## API déployée

- **Lien vers l'API** : [https://guim78.pythonanywhere.com](https://guim78.pythonanywhere.com)
- **Endpoint principal** : `/predict` (POST avec DataFrame JSON)

---

## Environnement Technique

- **Python**
- **Scikit-learn**, **imbalanced-learn**, **XGBoost**, **LightGBM**
- **MLflow** (log des modèles, métriques, comparaisons)
- **Joblib** (sérialisation)
- **Streamlit** (interface de test de l’API)
- **Flask** (API REST)
- **Evidently** (analyse du drift)
- **Git + GitHub** (gestion de version)
- **GitHub Actions** (déploiement continu)
- **PythonAnywhere** (hébergement cloud de l’API)

---

## Installation

```bash
git clone https://github.com/MaryG78/Mlflow_P7.git
cd Mlflow_P7
pip install -r requirements.txt
---

