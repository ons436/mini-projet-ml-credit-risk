"""
Application Streamlit : prédiction du risque de défaut de paiement (carte de crédit).

Lancer avec :   python -m streamlit run app.py
Fichiers nécessaires dans le même dossier :
    - app.py                         (ce fichier)
    - pretraitement.py               (fonction de prétraitement créée depuis le notebook)
    - modele_defaut_paiement.joblib  (modèle + prétraitement sauvegardés depuis le notebook)
"""
import joblib
import pandas as pd
import streamlit as st

from pretraitement import preparer   # même prétraitement que dans le notebook

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Risque de défaut de paiement", page_icon="💳", layout="wide")


# ---------------------------------------------------------------------------
# Chargement du modèle (une seule fois grâce au cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def charger_artefacts():
    return joblib.load("modele_defaut_paiement.joblib")


art = charger_artefacts()
SEUIL = art["seuil"]     # seuil de décision choisi dans le notebook (≈ 0,26)

# ---------------------------------------------------------------------------
# Dictionnaires : texte affiché en français -> valeur attendue par le modèle
# (le modèle a été entraîné avec les valeurs anglaises du dataset)
# ---------------------------------------------------------------------------
SEXE = {"Femme": "Female", "Homme": "Male"}
EDUCATION = {
    "Études supérieures (master / doctorat)": "Graduate school",
    "Université (licence)": "University",
    "Lycée": "High school",
    "Autre": "Others",
    "Inconnu": "Unknown",
}
MARIAGE = {
    "Célibataire": "Single",
    "Marié(e)": "Married",
    "Divorcé(e)": "Divorced",
    "Veuf / veuve": "Widowed",
    "Inconnu": "Unknown",
}
STATUT_PAIEMENT = {
    "Aucune consommation": "No consumption",
    "Payé en totalité à temps": "Paid duly",
    "Crédit renouvelable (paiement minimum)": "Revolving credit",
    "1 mois de retard": "1 month delay",
    "2 mois de retard": "2 months delay",
    "3 mois de retard": "3 months delay",
    "4 mois de retard": "4 months delay",
    "5 mois de retard": "5 months delay",
    "6 mois de retard": "6 months delay",
    "7 mois de retard": "7 months delay",
    "8 mois de retard ou plus": "8 months or more delay",
}

# Les 6 derniers mois : colonne PAY_x, BILL_AMTx, PAY_AMTx
# (PAY_0 = mois le plus récent, PAY_6 = le plus ancien ; il n'y a pas de PAY_1 dans le dataset)
COLS_PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
MOIS = ["Mois 1 (le plus récent)", "Mois 2", "Mois 3", "Mois 4", "Mois 5", "Mois 6 (le plus ancien)"]

# ---------------------------------------------------------------------------
# Titre
# ---------------------------------------------------------------------------
st.title("💳 Prédiction du risque de défaut de paiement")
st.caption(
    "Application de démonstration pédagogique (mini-projet Machine Learning, ISET Sfax). "
    "Modèle : Gradient Boosting optimisé, entraîné sur le jeu de données public « Default of Credit Card Clients » "
    "(Taïwan, 2005). Ce n'est pas un outil de décision de crédit réel."
)

# ---------------------------------------------------------------------------
# Formulaire de saisie
# ---------------------------------------------------------------------------
with st.form("formulaire_client"):
    st.subheader("1. Profil du client")
    c1, c2, c3, c4, c5 = st.columns(5)
    sexe = c1.selectbox("Sexe", list(SEXE))
    education = c2.selectbox("Niveau d'études", list(EDUCATION), index=1)
    mariage = c3.selectbox("Situation matrimoniale", list(MARIAGE))
    age = c4.number_input("Âge", min_value=18, max_value=100, value=34, step=1)
    limite = c5.number_input("Limite de crédit (TND)", min_value=0, value=12740, step=500)

    st.subheader("2. Historique des 6 derniers mois")
    tab_pay, tab_fact, tab_paie = st.tabs(
        ["Statut de remboursement", "Montant des factures (TND)", "Montant payé (TND)"]
    )

    statuts, factures, paiements = [], [], []
    with tab_pay:
        cols = st.columns(3)
        for i, mois in enumerate(MOIS):
            statuts.append(cols[i % 3].selectbox(mois, list(STATUT_PAIEMENT), index=1, key=f"pay{i}"))
    with tab_fact:
        cols = st.columns(3)
        for i, mois in enumerate(MOIS):
            factures.append(cols[i % 3].number_input(mois, value=2000, step=100, key=f"fact{i}"))
    with tab_paie:
        cols = st.columns(3)
        for i, mois in enumerate(MOIS):
            paiements.append(cols[i % 3].number_input(mois, min_value=0, value=150, step=50, key=f"paie{i}"))

    lancer = st.form_submit_button("Prédire le risque", type="primary")

# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------
if lancer:
    # 1. On construit UNE ligne avec exactement les colonnes d'origine du dataset
    ligne = {
        "LIMIT_BAL": limite,
        "SEX": SEXE[sexe],
        "EDUCATION": EDUCATION[education],
        "MARRIAGE": MARIAGE[mariage],
        "AGE": age,
    }
    for i, col in enumerate(COLS_PAY):
        ligne[col] = STATUT_PAIEMENT[statuts[i]]
    for i in range(6):
        ligne[f"BILL_AMT{i + 1}"] = factures[i]
        ligne[f"PAY_AMT{i + 1}"] = paiements[i]
    client = pd.DataFrame([ligne])

    # 2. Prétraitement identique au notebook, puis score interne du modèle
    X_client = preparer(client, art)
    score = float(art["modele"].predict_proba(X_client)[0, 1])   # utilisé seulement en interne

    # 3. Classification : le score est comparé au seuil choisi dans le notebook
    #    -> classe 1 = défaut de paiement, classe 0 = pas de défaut
    classe = int(score >= SEUIL)

    # 4. Affichage : uniquement la classe prédite (aucun pourcentage montré à l'utilisateur)
    st.divider()
    st.subheader("Résultat de la classification")
    r1, r2 = st.columns(2)
    r1.metric("Classe prédite", f"Classe {classe}")
    r2.metric("Catégorie", "Défaut de paiement" if classe == 1 else "Pas de défaut")

    if classe == 1:
        st.error(
            "⚠️ **Classe 1 : défaut de paiement.** "
            "Le client est classé dans la catégorie « risque de ne pas payer le mois suivant »."
        )
    else:
        st.success(
            "✅ **Classe 0 : pas de défaut.** "
            "Le client est classé dans la catégorie « paiera normalement le mois suivant »."
        )

    with st.expander("Comment lire ce résultat ?"):
        st.write(
            """
- Il s'agit d'un problème de **classification binaire** : le modèle range chaque client dans
  l'une de deux classes : **0 = pas de défaut**, **1 = défaut de paiement**.
- La décision est prise avec un **seuil ajusté dans le notebook** pour maximiser le F1
  (compromis précision / rappel). Comme les défauts sont rares (≈ 22 % des clients), le modèle est réglé
  pour détecter plus de défauts, quitte à avoir davantage de fausses alertes.
- Une classification « défaut » ne veut pas dire que le client fera défaut à coup sûr : le modèle se trompe,
  environ une alerte sur deux est une fausse alerte.
- Les montants sont à saisir en **dinars tunisiens (TND)**, comme dans le notebook.
"""
        )
