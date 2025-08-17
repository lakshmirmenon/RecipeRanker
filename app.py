import streamlit as st
from predict import predict_popularity, get_meta
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import base64
st.set_page_config(
    page_title="Recipe Popularity Predictor",
    page_icon="icons/cutlery.png",   # <-- path to your image
    layout="wide"
)

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_base64 = get_base64_image("icons/cutlery.png")
# Load meta + model for feature contributions
meta = get_meta()
class_names = meta["class_names"]
model = joblib.load("best_recipe_popularity_model.pkl")
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{image_base64}" width="60" style="margin-right: 15px;">
        <h1 style="margin: 0;">Recipe Popularity Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown("""
This app predicts how **popular** your recipe will be, as one of three categories:
- **Typical:** Score = 100 (the majority of recipes)
- **Popular:** Score 101–253 (higher than average)
- **Viral:** Score >253 (very rare!)
Enter your recipe details below 
""")

with st.form("prediction_form"):
    st.subheader(" Enter Recipe Details")
    recipe_name = st.text_input("Recipe Name", "Paneer Butter Masala")
    user_reputation = st.number_input("User Reputation", min_value=0, max_value=100000, value=1200)
    reply_count = st.number_input("Reply Count", min_value=0, value=10)
    thumbs_up = st.number_input("Thumbs Up ", min_value=0, value=50)
    thumbs_down = st.number_input("Thumbs Down ", min_value=0, value=2)
    stars = st.slider("Star Rating ", 0.0, 5.0, 4.5, 0.1)
    calories = st.number_input("Calories (kcal)", min_value=0, value=350)
    cooking_time = st.number_input("Cooking Time (minutes)", min_value=1, value=30)
    submitted = st.form_submit_button(" Predict Popularity")

if submitted:
    features = {
        "recipe_name": recipe_name.strip() if recipe_name.strip() else "Unknown",
        "user_reputation": user_reputation if user_reputation is not None else 0,
        "reply_count": reply_count if reply_count is not None else 0,
        "thumbs_up": thumbs_up if thumbs_up is not None else 0,
        "thumbs_down": thumbs_down if thumbs_down is not None else 0,
        "stars": stars if stars is not None else 0.0,
        "calories": calories if calories is not None else 0,
        "cooking_time": cooking_time if cooking_time is not None else 1,
    }
    idx, score_class, conf = predict_popularity(features)
    
    explain = {
        0: "Most recipes fall into the Typical category (score=100). To reach higher, boost engagement and thumbs up!",
        1: "Popular! Your recipe stands out above the typical ones. More engagement may make it go viral.",
        2: "Viral! Extremely rare. Only the top 10% recipes ever reach this class.",
    }
    st.success(f"### Prediction for **{recipe_name}**")
    st.metric("Predicted Popularity", score_class, f"Confidence: {100*conf:.1f}%")
    st.info(explain.get(idx, ""))

    # --- Probability Breakdown ---
    st.markdown("####  Probability Breakdown")
    probs = [0, 0, 0]
    probs[idx] = conf
    fig, ax = plt.subplots()
    bars = ax.bar(class_names, [100*p for p in probs], color=["green", "orange", "red"])
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{100*p:.1f}%", ha="center", fontweight="bold")
    st.pyplot(fig)

    # --- Feature Contribution Table ---
    st.markdown("#### 🔎 Feature Contributions")
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    X_input = model.named_steps['preprocessor'].transform(
        model.named_steps['feature_engineer'].transform(pd.DataFrame([features]))
    )
    contribs = importances * X_input[0]
    df_contrib = pd.DataFrame({
        "Feature": feature_names,
        "Value": X_input[0],
        "Importance": importances,
        "Contribution": contribs
    }).sort_values("Contribution", ascending=False).head(10)
    st.dataframe(df_contrib, use_container_width=True)

    # Original details
    st.markdown("#### Popularity class mapping:")
    st.write(f"""
    - 🟩 **Typical:** Score = 100 (covers {meta['pct_typical']:.1f}% of dataset)
    - 🟨 **Popular:** 101–253 (covers {meta['pct_popular']:.1f}%)
    - 🟥 **Viral (Top 10%):** >253 (covers {meta['pct_viral']:.1f}%)
    """)
    st.markdown("#### Confusion Matrix for validation set:")
    st.image("confusion_matrix.png")
