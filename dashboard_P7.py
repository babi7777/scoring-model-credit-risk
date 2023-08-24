#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import requests
import shap.explainers
import shap
import joblib
import io
import urllib.request
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

# Fonction pour charger le modèle depuis une URL
@st.cache_resource()
def load_model(model_url):
    response = requests.get(model_url)
    model = joblib.load(io.BytesIO(response.content))
    return model

def main():
    model_url = "https://github.com/babi7777/scoring-model-credit-risk/raw/main/modele_lgbm_over.pkl"
    
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
    client_data = get_client_preprocessed_data(selected_id) 
    model = load_model(model_url)

    if st.button("Prédire"):
                
        # Faire une prédiction avec le modèle
        prediction_proba = model.predict_proba(client_data.values.reshape(1, -1))[:, 1]
        prediction = "Refusé" if prediction_proba >= 0.435 else "Accepté"

        # Comparer la prédiction avec la vraie valeur de TARGET
        if prediction == "Refusé" and target_value == 1:
            prediction_check = "Correct (Vrai positif)"
        elif prediction == "Accepté" and target_value == 0:
            prediction_check = "Correct (Vrai négatif)"
        else:
            prediction_check = "Incorrect"
    
        # Afficher la prédiction
        st.subheader("Résultat de Prédiction")
        if prediction == "Accepté":
            st.write(f"Probabilité de Prédiction : {prediction_proba[0]:.4f}")
            st.markdown(f"<p style='font-size:18px; font-weight:bold; color:green;'>{prediction}</p>", unsafe_allow_html=True)
        else:
            st.write(f"Probabilité de Prédiction : {prediction_proba[0]:.4f}")
            st.markdown(f"<p style='font-size:18px; font-weight:bold; color:red;'>{prediction}</p>", unsafe_allow_html=True)
        st.write(f"Probabilité de Prédiction : {prediction_proba[0]:.4f}")
                
        st.write(f"Valeur de TARGET réelle : {target_value}")

        # Afficher les messages correspondants
        if prediction == "Refusé":
            st.write("La prédiction indique un refus de crédit.")
            if target_value == 1:
                st.write("La valeur de TARGET réelle confirme un défaut de paiement.")
            else:
                st.write("La valeur de TARGET réelle indique une non-défaillance de paiement.")
        else:
            st.write("La prédiction indique une acceptation de crédit.")
            if target_value == 0:
                st.write("La valeur de TARGET réelle confirme une non-défaillance de paiement.")
            else:
                st.write("La valeur de TARGET réelle indique un défaut de paiement.")
    
        # Calculer les valeurs SHAP pour le client        
        explainer = shap.TreeExplainer(model)        
    
        # Afficher l'interprétation SHAP des features        
        st.subheader("Interprétation SHAP des Features")        
        client_data_df = pd.DataFrame([client_data], columns=client_data.index)
        shap_values_client = explainer.shap_values(client_data_df)
        shap.force_plot(explainer.expected_value[1], shap_values_client[1][0], client_data, matplotlib=True)


if __name__ == '__main__':
    main()

