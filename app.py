#!/usr/bin/env python
# coding: utf-8

# In[42]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import requests
import shap
import joblib
import io
import urllib.request
from zipfile import ZipFile
plt.style.use('fivethirtyeight')

api_base_url = "35.181.54.91:5000"    
model_url = "https://github.com/babi7777/scoring-model-credit-risk/raw/main/modele_lgbm_over.pkl"

# Fonction pour obtenir les IDs clients disponibles depuis l'API
@st.cache_data()
def get_available_ids(api_base_url):
    ids_endpoint = f"{api_base_url}/api/clients"
    response = requests.get(ids_endpoint)
    available_ids_data = response.json()
    available_ids = available_ids_data["available_ids"]
    return available_ids

# Fonction pour obtenir les données d'un client depuis l'API
@st.cache_data()
def get_client_data(selected_id):
    client_data_endpoint = f"{api_base_url}/api/client/{selected_id}"
    response = requests.get(client_data_endpoint)
    client_data = response.json()
    return client_data

# Fonction pour obtenir les données prétraitées d'un client depuis l'API
@st.cache_data(hash_funcs={hash: hash})
def get_client_preprocessed_data(selected_id):
    api_url = f"{api_base_url}/api/client_preprocessed/{selected_id}"
    response = requests.get(api_url)    
    try:
        response_data = response.json()
        if isinstance(response_data, dict):
            client_preprocessed_data = response_data
            columns = list(client_preprocessed_data.keys())
            data = pd.DataFrame.from_dict(client_preprocessed_data, orient='index')
            return data
        else:
            raise ValueError("Invalid JSON response")
    except Exception as e:
        print("Error decoding JSON response:", e)
        return None
           
def load_model(model_url):    
    response = requests.get(model_url)
    model = joblib.load(io.BytesIO(response.content))
    return model 
    
def main():
        
    html_temp = """
    <div style="background-color: #475f4e ; padding:10px; border-radius:10px">
    <h1 style="color: #d9ae13; text-align:center">Dashboard de Prédiction de Crédit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)       
    
    # Changer la couleur du sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #89a791; /* Couleur de fond du sLidebar */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Interface Streamlit    
    st.sidebar.title("Informations Générales")
    # Personnaliser la couleur du sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #89a791;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # obtenir les ids
    available_ids = get_available_ids(api_base_url)
           
    # Sélectionner un ID client dans une liste déroulante
    selected_id = int(st.selectbox("Sélectionner un ID client", available_ids))
    
    # obtenir les informations du client
    client_info = get_client_data(selected_id)     
    
    # Afficher les informations du client
    st.sidebar.subheader("Informations du Client")
    st.sidebar.write("Age :", int(client_info["DAYS_BIRTH"] / -365), "ans")
    st.sidebar.write("Revenu total :", client_info["AMT_INCOME_TOTAL"])
    st.sidebar.write("Montant de crédit demandé :", client_info["AMT_CREDIT"])
    st.sidebar.write("Montant de l'annuité :", client_info["AMT_ANNUITY"])
    st.sidebar.write("Montant des biens pour le crédit :", client_info["AMT_GOODS_PRICE"])
    
    # charger les données et le modele en cache
    client_data_json = get_client_preprocessed_data(selected_id)       
    client_data = pd.DataFrame.from_dict(client_data_json)  
    model = load_model(model_url)

    
    # Obtenir la valeur de TARGET pour le client sélectionné depuis le JSON
    target_value = client_info["TARGET"]
    # Faire une prédiction avec le modèle
    response = requests.get(f"{api_base_url}/api/predict/{selected_id}")
    prediction_data = response.json()    
    prediction_proba = prediction_data["probability"]
    prediction_decision = prediction_data["decision"]

    # Afficher la prédiction
    st.subheader("Résultat de Prédiction")
    st.write(f"Probabilité de Prédiction : {prediction_proba:.4f}")
    if prediction_decision == "Accepted":
        st.markdown(f"<p style='font-size:18px; font-weight:bold; color:green;'>{prediction_decision}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size:18px; font-weight:bold; color:red;'>{prediction_decision}</p>", unsafe_allow_html=True)
               
    # Afficher les messages correspondants
    if prediction_decision == "Denied":
        st.write("La prédiction indique un refus de crédit.")            
    else:
        st.write("La prédiction indique une acceptation de crédit.")                      
                

if __name__ == '__main__':
    main()