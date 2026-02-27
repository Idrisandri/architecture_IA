from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Wine Quality API 🍷")

# ✅ CORS pour Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # remplace par ton URL Netlify en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Charger les modèles
rf_model = joblib.load("model/random_forest_wine.pkl")
scaler   = joblib.load("model/scaler_wine.pkl")

# ✅ Connexion Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# ✅ Colonnes attendues par le modèle
COLONNES = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# ✅ Schema des données entrantes
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

# ✅ Route de test
@app.get("/")
def home():
    return {"message": "Wine Quality API is running 🍷"}

# ✅ Route principale : prédiction
@app.post("/predict")
def predict(wine: WineInput):
    try:
        # 1. Préparer les données
        data = [[
            wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid,
            wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide,
            wine.total_sulfur_dioxide, wine.density, wine.ph,
            wine.sulphates, wine.alcohol
        ]]
        df = pd.DataFrame(data, columns=COLONNES)

        # 2. Scaler + Prédiction
        df_scaled  = scaler.transform(df)
        prediction = rf_model.predict(df_scaled)[0]
        label      = "Bon" if prediction == 1 else "Mauvais"

        # 3. Update Supabase — cherche la ligne sans quality_label
        supabase.table("wine_predictions").update(
            {"quality_label": label}
        ).match({
            "fixed_acidity": wine.fixed_acidity,
            "volatile_acidity": wine.volatile_acidity,
            "quality_label": None
        }).execute()

        return {
            "prediction": int(prediction),
            "quality_label": label,
            "message": f"Ce vin est {label} 🍷" if label == "Bon" else f"Ce vin est {label} 😐"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Route : récupérer tous les vins depuis Supabase
@app.get("/wines")
def get_wines():
    response = supabase.table("wine_predictions").select("*").execute()
    return response.data