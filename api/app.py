import os
# Fix pour éviter l'erreur LOKY sur PythonAnywhere
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd


app = Flask(__name__)

# Charge le modèle sauvegardé
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
        df = df.astype("float32")  # Assure la compatibilité avec LightGBM

        prediction = model.predict_proba(df)[:, 1]  # 💡 Prédiction de la proba
        return jsonify({"predictions": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)