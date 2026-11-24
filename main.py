
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pretraitement import preparer


# 1. Initialisation de FastAPI
app = FastAPI(
    title="API de prédiction du risque de défaut",
    description="API Machine Learning utilisant Gradient Boosting",
    version="1.0.0"
)


# 2. Chargement du modèle et des artefacts
BASE_DIR = Path(__file__).resolve().parent

art = joblib.load(BASE_DIR / "modele_defaut_paiement.joblib")
SEUIL = float(art["seuil"])


# 3. Correspondance des catégories
SEXE = {
    "Femme": "Female",
    "Homme": "Male"
}

EDUCATION = {
    "Études supérieures (master / doctorat)": "Graduate school",
    "Université (licence)": "University",
    "Lycée": "High school",
    "Autre": "Others",
    "Inconnu": "Unknown"
}

MARIAGE = {
    "Célibataire": "Single",
    "Marié(e)": "Married",
    "Divorcé(e)": "Divorced",
    "Veuf / veuve": "Widowed",
    "Inconnu": "Unknown"
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
    "8 mois de retard ou plus": "8 months or more delay"
}

COLS_PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


# 4. Définition des données reçues
class Client(BaseModel):
    sexe: str
    education: str
    mariage: str
    age: int = Field(ge=18, le=100)
    limite: float = Field(ge=0)

    statuts: list[str] = Field(min_length=6, max_length=6)
    factures: list[float] = Field(min_length=6, max_length=6)
    paiements: list[float] = Field(min_length=6, max_length=6)


# 5. Route d'accueil
@app.get("/")
def accueil():
    return {
        "message": "API de prédiction du risque de défaut opérationnelle",
        "documentation": "/docs"
    }


# 6. Route de prédiction
@app.post("/predict")
def predire(client: Client):

    # Vérification des catégories
    if client.sexe not in SEXE:
        raise HTTPException(400, "Sexe invalide")

    if client.education not in EDUCATION:
        raise HTTPException(400, "Niveau d'études invalide")

    if client.mariage not in MARIAGE:
        raise HTTPException(400, "Situation matrimoniale invalide")

    if any(s not in STATUT_PAIEMENT for s in client.statuts):
        raise HTTPException(400, "Statut de paiement invalide")

    # Construction des données brutes
    ligne = {
        "LIMIT_BAL": client.limite,
        "SEX": SEXE[client.sexe],
        "EDUCATION": EDUCATION[client.education],
        "MARRIAGE": MARIAGE[client.mariage],
        "AGE": client.age
    }

    for i, col in enumerate(COLS_PAY):
        ligne[col] = STATUT_PAIEMENT[client.statuts[i]]

    for i in range(6):
        ligne[f"BILL_AMT{i + 1}"] = client.factures[i]
        ligne[f"PAY_AMT{i + 1}"] = client.paiements[i]

    try:
        # Prétraitement identique au notebook
        df_client = pd.DataFrame([ligne])
        X_client = preparer(df_client, art)

        # Prédiction
        proba = float(
            art["modele"].predict_proba(X_client)[0, 1]
        )

        # Décision avec le seuil choisi
        decision = (
            "ALERTE"
            if proba >= SEUIL
            else "RISQUE FAIBLE"
        )

        return {
            "probabilite_defaut": round(proba, 4),
            "seuil": SEUIL,
            "decision": decision
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )
