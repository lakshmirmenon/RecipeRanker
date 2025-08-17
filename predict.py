import joblib
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer

# Load model and meta
model = joblib.load("best_recipe_popularity_model.pkl")
meta = joblib.load("model_meta.pkl")

def get_meta():
    return meta

def predict_popularity(features_dict):
    raw_input = {
        "recipe_name": features_dict.get("recipe_name", "Unknown"),
        "user_reputation": features_dict.get("user_reputation", 0),
        "reply_count": features_dict.get("reply_count", 0),
        "thumbs_up": features_dict.get("thumbs_up", 0),
        "thumbs_down": features_dict.get("thumbs_down", 0),
        "stars": features_dict.get("stars", 0.0),
        "calories": features_dict.get("calories", 0),
        "cooking_time": features_dict.get("cooking_time", 1),
    }
    X = pd.DataFrame([raw_input])
    class_idx = int(model.predict(X)[0])

    # Always return probabilities for [0, 1, 2]
    probs = np.zeros(3)
    model_classes = model.named_steps['clf'].classes_
    pred_proba = model.predict_proba(X)[0]   # ✅ fixed
    for idx, cls in enumerate(model_classes):
        probs[int(cls)] = pred_proba[idx]

    score_class = meta["class_names"][class_idx]
    score_class_proba = probs[class_idx]
    return class_idx, score_class, float(score_class_proba)
