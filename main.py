from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import json
import os
from datetime import datetime

app = FastAPI()

# CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Load models
model_attack = joblib.load('model_attack.pkl')
scaler = joblib.load('scaler.pkl')
model_relief = joblib.load('model_relief.pkl')
scaler_relief = joblib.load('scaler_relief.pkl')
le_relief = joblib.load('le_relief.pkl')
TOP_FEATURES = joblib.load('feature_cols.pkl')

with open('model_meta.json', 'r') as f:
    meta = json.load(f)

print(f"✅ All models loaded successfully")
print(f"   Features: {TOP_FEATURES}")

# Request model matching your MigraineInput class
class PredictionInput(BaseModel):
    Stress_Level: int
    Sleep_Hours: float
    Water_Intake_L: float
    Air_Quality_Index: int
    Screen_Time_Hours: float
    Sleep_Quality: int
    Depression_Level: int
    Anxiety_Level: int
    Caffeine_Intake: int
    Skipped_Meals: int

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

# Root endpoint
@app.get("/")
async def root():
    return {"status": "ok", "message": "Migraine Prediction API v2.0 is running"}

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert to numpy array with correct order
        features = np.array([[
            input_data.Stress_Level,
            input_data.Sleep_Hours,
            input_data.Water_Intake_L,
            input_data.Air_Quality_Index,
            input_data.Screen_Time_Hours,
            input_data.Sleep_Quality,
            input_data.Depression_Level,
            input_data.Anxiety_Level,
            input_data.Caffeine_Intake,
            input_data.Skipped_Meals
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get attack probability
        attack_prob = model_attack.predict_proba(features_scaled)[0][1]
        attack_likely = bool(attack_prob >= 0.4)  # Convert to Python bool
        
        # Calculate risk score (0-100)
        risk_score = int(attack_prob * 100)
        
        # Determine severity based on probability
        if attack_prob >= 0.75:
            severity = "Severe"
            warning_level = "HIGH"
            warning_message = "High risk of severe migraine attack. Take preventive measures immediately."
        elif attack_prob >= 0.45:
            severity = "Moderate"
            warning_level = "MEDIUM"
            warning_message = "Moderate risk detected. Consider taking action."
        else:
            severity = "Mild"
            warning_level = "LOW"
            warning_message = "Low risk. Continue monitoring your symptoms."
        
        # Get relief recommendation
        relief_scaled = scaler_relief.transform(features)
        relief_encoded = model_relief.predict(relief_scaled)[0]
        relief = le_relief.inverse_transform([relief_encoded])[0]
        
        # Generate relief tips
        relief_tips = get_relief_tips(relief, severity)
        
        # Get top triggers (features with highest values)
        feature_values = dict(zip(TOP_FEATURES, features[0]))
        sorted_triggers = sorted(feature_values.items(), key=lambda x: x[1], reverse=True)[:3]
        top_triggers = [{"feature": f, "weight": float(v/10)} for f, v in sorted_triggers]  # Convert to float
        
        # Return response matching Flutter's expected format
        response = {
            "risk_score": risk_score,
            "attack_likely": attack_likely,
            "attack_prob": float(attack_prob),  # Convert to float
            "warning_level": warning_level,
            "warning_message": warning_message,
            "severity": severity,
            "relief": relief,
            "relief_tips": relief_tips,
            "top_triggers": top_triggers,
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert any remaining numpy types to Python native types
        response = {k: convert_to_serializable(v) for k, v in response.items()}
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_relief_tips(relief, severity):
    """Generate relief tips based on recommendation"""
    tips_map = {
        "Sleep": ["Lie down in a quiet room", "Aim for 7-9 hours of sleep", "Avoid screens before bed"],
        "Meditation": ["Try 10 min deep breathing", "Use a guided relaxation app", "Reduce stimulation around you"],
        "Hydration": ["Drink 2 glasses of water now", "Avoid caffeine and alcohol", "Add electrolytes if needed"],
        "Medication": ["Take prescribed medication early", "Rest after taking medication", "Track medication timing"],
        "Dark Room": ["Move to a dark, quiet room", "Apply a cold compress to forehead", "Minimise noise exposure"],
        "Cold Compress": ["Apply cold pack to forehead", "Keep eyes closed and rest", "Stay in a cool environment"],
        "Rest": ["Rest in a comfortable position", "Avoid bright lights", "Stay hydrated"]
    }
    return tips_map.get(relief, tips_map["Rest"])