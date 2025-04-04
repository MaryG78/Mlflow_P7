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
            'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN': 0.5714285969734192,
            'BURO_DAYS_CREDIT_MIN': -2359.0,
            'APPROVED_APP_CREDIT_PERC_MAX': 1.0587959289550781,
            'CC_CNT_DRAWINGS_ATM_CURRENT_VAR': 3.4945051670074463,
            'INSTAL_DPD_MEAN': 0.0,
            'APPROVED_DAYS_DECISION_MIN': -2133.0,
            'PREV_DAYS_DECISION_MIN': -2133.0,
            'INSTAL_DAYS_ENTRY_PAYMENT_SUM': -37180.0,
            'INSTAL_DBD_SUM': 423.0,
            'EXT_SOURCE_1_EXT_SOURCE_3': 0.23956459760665894,
            'EXT_SOURCE_3_DAYS_BIRTH': -7660.69482421875,
            'EXT_SOURCE_1_x': 0.3845820128917694,
            'BURO_DAYS_CREDIT_ENDDATE_MEAN': 1633.5999755859375,
            'BURO_DAYS_CREDIT_ENDDATE_MIN': -1996.0,
            'INSTAL_DAYS_ENTRY_PAYMENT_MEAN': -652.2807006835938,
            'EXT_SOURCE_2_DAYS_BIRTH': -3561.166015625,
            'EXT_SOURCE_2_EXT_SOURCE_3': 0.18038125336170197,
            'PREV_AMT_ANNUITY_MIN': 3455.10009765625,
            'INSTAL_PAYMENT_PERC_MEAN': 1.0361127853393555,
            'INSTAL_AMT_PAYMENT_MAX': 31500.0
        }


        df = pd.DataFrame([input_row])  # Une seule ligne
        df = df.astype("float32")       # Cast obligatoire pour ton modèle

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
