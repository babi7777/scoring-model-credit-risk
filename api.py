#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import io
from zipfile import ZipFile

app = Flask(__name__)

def load_data():
    data_url = "https://github.com/babi7777/scoring-model-credit-risk/raw/main/X_test.zip"
    response = requests.get(data_url)
    with io.BytesIO(response.content) as zip_file:
        with ZipFile(zip_file, "r") as z:
            data = pd.read_csv(z.open('X_test.csv'), index_col='SK_ID_CURR', encoding='utf-8')
            return data
            
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

# Chargement initial des données et du modèle
data = load_data()
raw_data = load_raw_data()
model = load_model()

@app.route('/api/clients', methods=['GET'])
def get_clients():
    return jsonify({"available_ids": data.index.tolist()})

@app.route('/api/client/<int:id>', methods=['GET'])
def get_client_data(id):
    if id in raw_data.index:
        client_raw_data = raw_data.loc[id].to_dict()
        return jsonify(client_raw_data)
    else:
        return jsonify({"error": "Client ID not found"}), 404

@app.route('/api/client_preprocessed/<int:id>', methods=['GET'])
def get_client_preprocessed_data(id):
    if id in data.index:
        client_data_preprocessed = data.loc[id].to_dict()
        return jsonify(client_data_preprocessed)
    else:
        return jsonify({"error": "Client ID not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

