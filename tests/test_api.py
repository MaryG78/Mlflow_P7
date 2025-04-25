import sys
import os
import unittest
import json
import pandas as pd

# Ajout du chemin racine pour que `api.app` soit trouvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn("API de Scoring", response.get_data(as_text=True))

    def test_predict(self):
        # Ligne de test avec les 20 features du modèle
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


        df = pd.DataFrame([input_row])  
        df = df.astype("float32")       

        test_data = {"dataframe_split": df.to_dict(orient="split")}

        response = self.app.post('/predict',
                                 json=test_data,
                                 content_type='application/json')

        print("Code HTTP :", response.status_code)
        print("Contenu brut :", response.data.decode())

        self.assertEqual(response.status_code, 200)
        self.assertIn('predictions', json.loads(response.data))


if __name__ == "__main__":
    unittest.main()
