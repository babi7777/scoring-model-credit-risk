#!/usr/bin/env python
# coding: utf-8

# In[43]:


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

api_base_url = "http://35.181.54.91:5000"    
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
    client_preprocessed_data = response.json()
    columns = list(client_preprocessed_data.keys())  
    data = pd.DataFrame.from_dict(client_preprocessed_data, orient='index')
    return data
           
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
                
    if st.button("Interprétation"):
        # Calculer les valeurs SHAP pour le client        
        explainer = shap.TreeExplainer(model)        
    
        # Afficher l'interprétation SHAP des features        
        st.subheader("Interprétation SHAP des Features")    
    
        shap_values_client = explainer.shap_values(client_data.T)  
        shap_value = shap_values_client[1][0]      
        shap.force_plot(explainer.expected_value[1], shap_value, client_data.T, matplotlib=True)
       
    # Fonction pour obtenir les données prétraitées d'un nouveau client    
    def load_new_data():
        new_url = "https://github.com/babi7777/scoring-model-credit-risk/raw/main/data_test.zip"
        response = requests.get(new_url)
    
        with io.BytesIO(response.content) as zip_file:
            with ZipFile(zip_file, "r") as z:
                z.extractall()  # Extraction du contenu du fichier ZIP
                new_data = pd.read_csv('data_test.csv', index_col='SK_ID_CURR', encoding='utf-8')
                new_ids = new_data.index.tolist()
    
        return new_data, new_ids
    
    # charger l'échantillon des données pour les nouveaux clients
    new_data, new_ids = load_new_data()
    
    # Sélectionner un ID d'un nouveau client dans une liste déroulante
    selected_new_id = st.selectbox("Sélectionner un ID d'un nouveau client", new_ids)
    
    # Ajouter un bouton pour prédire de nouveaux clients en utilisant le modèle
    if st.button("Prédire pour Nouveaux Clients"):       
    
        if selected_new_id in new_ids:
            new_client_data = new_data.loc[selected_new_id]
            new_prediction_proba = model.predict_proba(new_client_data.values.reshape(1, -1))[:, 1]
            new_prediction = "Denied" if new_prediction_proba >= 0.435 else "Accepted"
        
            st.subheader("Prédiction : ")
            st.write(f"Probabilité de faillite : {new_prediction_proba[0]:.4f}")
            if new_prediction == "Accepted":
                st.markdown(f"<p style='font-size:18px; font-weight:bold; color:green;'>{new_prediction}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='font-size:18px; font-weight:bold; color:red;'>{new_prediction}</p>", unsafe_allow_html=True)
        else:
            st.warning("ID de client nouveau invalide. Veuillez sélectionner un ID valide.")

if __name__ == '__main__':
    main()

