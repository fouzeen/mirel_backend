from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
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
 
# Strong helper function to convert ANY numpy type to Python native (NumPy 2.0 compatible)
def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    type_name = type(obj).__name__
    
    if type_name in ('bool_', 'bool'):
        return bool(obj)
    if type_name in ('int_', 'int8', 'int16', 'int32', 'int64', 'integer'):
        return int(obj)
    if type_name in ('float_', 'float16', 'float32', 'float64', 'floating'):
        return float(obj)
    if type_name == 'ndarray':
        return obj.tolist()
    
    if isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj
 
# ✅ FIX: Use absolute paths so Render can always find the .pkl files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
model_attack = joblib.load(os.path.join(BASE_DIR, 'model_attack.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
model_relief = joblib.load(os.path.join(BASE_DIR, 'model_relief.pkl'))
scaler_relief = joblib.load(os.path.join(BASE_DIR, 'scaler_relief.pkl'))
le_relief = joblib.load(os.path.join(BASE_DIR, 'le_relief.pkl'))
TOP_FEATURES = joblib.load(os.path.join(BASE_DIR, 'feature_cols.pkl'))
 
with open(os.path.join(BASE_DIR, 'model_meta.json'), 'r') as f:
    meta = json.load(f)
 
print(f"✅ All models loaded successfully")
print(f"   Features: {TOP_FEATURES}")
 
# Request model
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
 
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}
 
@app.get("/")
async def root():
    return {"status": "ok", "message": "Migraine Prediction API v2.0 is running"}
 
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        features = np.array([[
            float(input_data.Stress_Level),
            float(input_data.Sleep_Hours),
            float(input_data.Water_Intake_L),
            float(input_data.Air_Quality_Index),
            float(input_data.Screen_Time_Hours),
            float(input_data.Sleep_Quality),
            float(input_data.Depression_Level),
            float(input_data.Anxiety_Level),
            float(input_data.Caffeine_Intake),
            float(input_data.Skipped_Meals)
        ]])
        
        features_scaled = scaler.transform(features)
        
        attack_prob_array = model_attack.predict_proba(features_scaled)[0]
        attack_prob = float(attack_prob_array[1])
        attack_likely = attack_prob >= 0.4
        
        risk_score = int(attack_prob * 100)
        
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
        
        relief_scaled = scaler_relief.transform(features)
        relief_encoded = model_relief.predict(relief_scaled)[0]
        relief = le_relief.inverse_transform([relief_encoded])[0]
        
        relief_tips = get_relief_tips(relief, severity)
        
        feature_values = dict(zip(TOP_FEATURES, features[0]))
        sorted_triggers = sorted(feature_values.items(), key=lambda x: x[1], reverse=True)[:3]
        top_triggers = [{"feature": str(f), "weight": float(v/10)} for f, v in sorted_triggers]
        
        response = {
            "risk_score": risk_score,
            "attack_likely": attack_likely,
            "attack_prob": attack_prob,
            "warning_level": warning_level,
            "warning_message": warning_message,
            "severity": severity,
            "relief": str(relief),
            "relief_tips": relief_tips,
            "top_triggers": top_triggers,
            "timestamp": datetime.now().isoformat()
        }
        
        response = convert_to_serializable(response)
        return response
        
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
 
def get_relief_tips(relief, severity):
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
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
 