import numpy as np
import pandas as pd

def preparer(df_brut, art):
    """Transforme des clients bruts (colonnes d'origine, montants en TND)
    en variables prêtes pour le modèle. Refait les mêmes étapes que le notebook."""
    X = df_brut[art["colonnes_brutes"]].copy()

    # 1. Imputation (médiane / valeur la plus fréquente apprises sur le train)
    X[art["cols_num"]] = art["imp_num"].transform(X[art["cols_num"]])
    X[art["cols_cat"]] = art["imp_cat"].transform(X[art["cols_cat"]])

    # 2. Encodage ordonné des colonnes PAY (texte -> -2, -1, 0, 1 ... 8)
    for c in art["cols_pay"]:
        X[c] = X[c].map(art["ordre_pay"])

    # 3. One-hot "à la main" : on crée exactement les colonnes vues à l'entraînement
    for col in art["colonnes_finales"]:
        if "_" in col and col.split("_", 1)[0] in art["cols_onehot"]:
            var, modalite = col.split("_", 1)
            X[col] = (X[var] == modalite).astype(int)
    X = X.drop(columns=art["cols_onehot"])

    # 4. Mêmes colonnes, dans le même ordre qu'à l'entraînement
    X = X[art["colonnes_finales"]]

    # 5. Normalisation (moyenne et écart-type du train)
    X[art["cols_a_normaliser"]] = art["scaler"].transform(X[art["cols_a_normaliser"]])
    return X
