import unittest
import json
import pandas as pd
from api.app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn("API de Scoring", response.get_data(as_text=True))

    def test_predict(self):
        import pandas as pd

        # Une seule ligne avec les bonnes colonnes
        input_row = {
            "EXT_SOURCE_2_EXT_SOURCE_3": 0.18038125336170197,
            "EXT_SOURCE_2_DAYS_BIRTH": -3561.166015625,
            "EXT_SOURCE_2_x": 0.2895727753639221,
            "EXT_SOURCE_1_EXT_SOURCE_2": 0.11136448383331299,
            "EXT_SOURCE_2_y": 0.2895727753639221,
            "EXT_SOURCE_1_EXT_SOURCE_3": 0.23956459760665894,
            "EXT_SOURCE_2_2": 0.08385239541530609,
            "EXT_SOURCE_3_x": 0.6229220032691956,
            "EXT_SOURCE_3_DAYS_BIRTH": -7660.69482421875,
            "EXT_SOURCE_3_y": 0.6229220032691956,
            "EXT_SOURCE_3_2": 0.38803181052207947,
            "BURO_DAYS_CREDIT_MEAN": -1175.800048828125,
            "EXT_SOURCE_1_x": 0.3845820128917694,
            "EXT_SOURCE_1_y": 0.3845820128917694,
            "EXT_SOURCE_1_2": 0.14790332317352295,
            "BURO_DAYS_CREDIT_MAX": -592.0,
            "INSTAL_DPD_MEAN": 0.0,
            "BURO_DAYS_CREDIT_UPDATE_MEAN": -584.4000244140625,
            "INSTAL_PAYMENT_PERC_MEAN": 1.0361127853393555,
            "INSTAL_AMT_INSTALMENT_SUM": 519160.8125,
            "BURO_DAYS_CREDIT_ENDDATE_MEAN": 1633.5999755859375,
            "APPROVED_DAYS_DECISION_MIN": -2133.0,
            "INSTAL_AMT_PAYMENT_SUM": 550660.8125,
            "PREV_DAYS_DECISION_MIN": -2133.0,
            "INSTAL_PAYMENT_DIFF_MEAN": -552.631591796875,
            "INSTAL_AMT_PAYMENT_MAX": 31500.0,
            "INSTAL_DAYS_ENTRY_PAYMENT_MEAN": -652.2807006835938,
            "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0.5714285969734192,
            "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -37180.0,
            "APPROVED_DAYS_DECISION_MEAN": -1178.199951171875,
            "APPROVED_AMT_ANNUITY_MIN": 3455.10009765625,
            "BURO_DAYS_CREDIT_MIN": -2359.0,
            "APPROVED_APP_CREDIT_PERC_MAX": 1.0587959289550781,
            "INSTAL_DBD_MEAN": 7.4210524559021,
            "PREV_AMT_ANNUITY_MIN": 3455.10009765625,
            "INSTAL_DBD_SUM": 423.0,
            "CC_CNT_DRAWINGS_ATM_CURRENT_VAR": 3.4945051670074463,
            "PREV_DAYS_DECISION_MEAN": -1178.199951171875,
            "BURO_DAYS_CREDIT_ENDDATE_MIN": -1996.0,
            "INSTAL_AMT_INSTALMENT_MAX": 19147.814453125
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
