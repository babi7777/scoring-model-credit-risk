#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pandas as pd
from zipfile import ZipFile
import joblib

app = Flask(__name__)

# Charger les données et le modèle en cache
def load_data():
    z = ZipFile("X_test.zip")
    data = pd.read_csv(z.open('X_test.csv'), index_col='SK_ID_CURR', encoding='utf-8')
    available_ids = data.index.tolist()
    return data, available_ids

def load_model():
    model_filename = 'scoring-modl-credit-risk/scoring-model-credit-risk/modele_lgbm_over.pkl'
    model = joblib.load(model_filename)
    return model

data, available_ids = load_data()
model = load_model()

# Point API pour fournir la liste des ID clients
@app.route('/api/clients', methods=['GET'])
def get_clients():
    return jsonify({"available_ids": available_ids})

# Point API pour fournir les données d'un client (ID)
@app.route('/api/client/<int:id>', methods=['GET'])
def get_client_data(id):
    if id in available_ids:
        client_data = data.loc[id].to_dict()
        return jsonify(client_data)
    else:
        return jsonify({"error": "Client ID not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

