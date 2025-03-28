import os
# Fix pour éviter l'erreur LOKY sur PythonAnywhere
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from flask import Flask, request, jsonify
import joblib
import mlflow.pyfunc
import pandas as pd


app = Flask(__name__)

# Chargement du modèle sauvegardé avec joblib
model_path = os.path.join(os.path.dirname(__file__), "..", "best_model.joblib")
model = joblib.load(model_path)

@app.route("/", methods=["GET"])
def home():
    return "API de Scoring avec modèle joblib"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(**data["dataframe_split"])
        df = df.astype("float32")  # 💡 Assure la compatibilité avec LightGBM

        # 🔮 Prédiction de la probabilité de remboursement
        prediction = model.predict_proba(df)[:, 1]
        return jsonify({"predictions": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🔧 Cette partie n'est utilisée que pour les tests en local
if __name__ == "__main__":
    app.run(debug=True)