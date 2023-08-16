#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pandas as pd
import multiprocessing
from zipfile import ZipFile
import joblib
import requests
import io

app = Flask(__name__)

# Charger les données et le modèle en cache
def load_data():
    data_url = "https://github.com/babi7777/scoring-model-credit-risk/raw/main/X_test.zip"
    response = requests.get(data_url)
    with io.BytesIO(response.content) as zip_file:
        with ZipFile(zip_file, "r") as z:
            data = pd.read_csv(z.open('X_test.csv'), index_col='SK_ID_CURR', encoding='utf-8')
            available_ids = data.index.tolist()
            return data, available_ids
            
def load_raw_data():    
    data_brut_url = "https://github.com/babi7777/scoring-model-credit-risk/raw/main/X_test_brut.zip"
    response = requests.get(data_brut_url)
    with io.BytesIO(response.content) as zip_file:
        with ZipFile(zip_file, "r") as z:
            raw_data = pd.read_csv(z.open('X_test_brut.csv'), index_col='SK_ID_CURR', encoding='utf-8')
            return raw_data

def load_model():
    model_url = "https://github.com/babi7777/scoring-model-credit-risk/raw/main/modele_lgbm_over.pkl"
    response = requests.get(model_url)
    model = joblib.load(io.BytesIO(response.content))
    return model
def main():
    process_data = multiprocessing.Process(target=load_data)
    process_model = multiprocessing.Process(target=load_model)

    process_data.start()
    process_model.start()

    process_data.join()
    process_model.join()
    
data, available_ids = load_data()
raw_data = load_raw_data()
model = load_model()

# Point API pour fournir la liste des ID clients
@app.route('/api/clients', methods=['GET'])
def get_clients():
    return jsonify({"available_ids": available_ids})

# Point API pour fournir les données d'un client (ID)
@app.route('/api/client/<int:id>', methods=['GET'])
def get_client_data(id):
    if id in available_ids:
        client_raw_data = raw_data.loc[id].to_dict()
        return jsonify(client_raw_data)
    else:
        return jsonify({"error": "Client ID not found"}), 404

# Point API pour fournir les données prétraitées d'un client spécifique
@app.route('/api/client_preprocessed/<int:id>', methods=['GET'])
def get_client_preprocessed_data(id):
    if id in available_ids:
        client_data_preprocessed = data.loc[id].to_dict()
        return jsonify(client_data_preprocessed)
    else:
        return jsonify({"error": "Client ID not found"}), 404

# Point API pour effectuer une prédiction avec le modèle
@app.route('/api/predict/<int:id>', methods=['GET'])
def predict(id):
    if id in available_ids:
        client_data = data.loc[id]  # Obtenir les données prétraitées du client
        prediction_proba = model.predict_proba(client_data.values.reshape(1, -1))[:, 1]
        prediction = "Refusé" if prediction_proba >= 0.435 else "Accepté"
        return jsonify({"prediction": prediction})
    else:
        return jsonify({"error": "Client ID not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

