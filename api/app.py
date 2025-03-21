from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

app = Flask(__name__)

# Chargement ldu modèle MLflow
model_path = os.environ.get("MODEL_PATH", "./mlflow_model")
model = mlflow.pyfunc.load_model(model_path)

@app.route("/", methods=["GET"])
def home():
    return "API de Scoring en Production sur PythonAnywhere "

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(**data["dataframe_split"])

        # FORCER LA CONVERSION en float32 dans l'API
        df = df.astype("float32")

        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)