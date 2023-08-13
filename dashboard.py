#!/usr/bin/env python
# coding: utf-8

# In[33]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import shap
from zipfile import ZipFile
plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')

# Décorateur pour charger les données en cache
@st.cache_data
def load_data():
    z = ZipFile("X_test.zip")
    data = pd.read_csv(z.open('X_test.csv'), index_col='SK_ID_CURR', encoding='utf-8')
    available_ids = data.index.tolist()  # Ajoutez cette ligne pour récupérer les IDs disponibles
    return data, available_ids

@st.cache_data
def load_raw_data():
    z = ZipFile("X_test_brut.zip")
    raw_data = pd.read_csv(z.open('X_test_brut.csv'), index_col='SK_ID_CURR', encoding='utf-8')
    return raw_data

@st.cache_data
def load_model():
    model_filename = 'scoring-modl-credit-risk/scoring-model-credit-risk/modele_lgbm_over.pkl'
    model = joblib.load(model_filename)
    return model

def main():
    html_temp = """
    <div style="background-color: #475f4e ; padding:10px; border-radius:10px">
    <h1 style="color: #d9ae13; text-align:center">Dashboard de Prédiction de Crédit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Charger les données et le modèle en cache
    data, available_ids = load_data()
    raw_data = load_raw_data()
    model = load_model()
    # Changer la couleur du sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #89a791; /* Couleur de fond du sidebar */
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
    
    # Sélectionner un ID client dans une liste déroulante
    selected_id = st.selectbox("Sélectionner un ID client", available_ids)
    
    # Obtenir les données correspondant à l'ID sélectionné
    client_raw_data = raw_data.loc[selected_id]

    # Afficher les informations du client
    st.sidebar.subheader("Informations du Client")
    st.sidebar.write("Age :", int(client_raw_data["DAYS_BIRTH"] / -365), "ans")
    st.sidebar.write("Revenu total :", client_raw_data["AMT_INCOME_TOTAL"])
    st.sidebar.write("Montant de crédit demandé :", client_raw_data["AMT_CREDIT"])
    st.sidebar.write("Montant de l'annuité :", client_raw_data["AMT_ANNUITY"])
    st.sidebar.write("Montant des biens pour le crédit :", client_raw_data["AMT_GOODS_PRICE"])
    
    if st.button("Prédire"):
        # Obtenir les données prétraitées correspondant à l'ID sélectionné pour effectuer la prédiction
        client_data = data.loc[selected_id]
        # Obtenir la valeur de TARGET pour le client sélectionné à partir des données brutes
        target_value = raw_data.loc[selected_id, "TARGET"]
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

