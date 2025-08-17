import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from feature_engineering import FeatureEngineer

print(" Loading dataset...")
df = pd.read_csv("reviews.csv")
print(f" Dataset loaded with shape: {df.shape}")

target = "best_score"
base_num_features = [
    "user_reputation", "reply_count", "thumbs_up",
    "thumbs_down", "stars", "calories", "cooking_time"
]
cat_features = ["recipe_name"]
needed = cat_features + base_num_features + [target]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=[target]).copy()
# Bin target: 0="Typical", 1="Popular", 2="Viral"
bins = [0, 100, 253, np.inf]
labels = [0, 1, 2]
df["popularity_class"] = pd.cut(df[target], bins=bins, labels=labels, right=True, include_lowest=True).astype(int)

print("Class label counts:", dict(df["popularity_class"].value_counts()))

X = df[cat_features + base_num_features]
y = df["popularity_class"]

print(" Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = base_num_features + ["thumbs_ratio", "engagement", "stars_weighted"]
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ]
)

print(" Training classifier...")
model = Pipeline(steps=[
    ("feature_engineer", FeatureEngineer()),
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# FORCE all 3 classes (0, 1, 2) are always present for predict_proba
all_classes = np.array([0, 1, 2])
clf = model.named_steps["clf"]
if len(clf.classes_) < 3:
    clf.classes_ = all_classes

print(" Training complete. Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "best_recipe_popularity_model.pkl")

# Save meta with class maps
class_names = ["Typical", "Popular", "Viral"]
meta = {
    "class_bins": bins,
    "class_labels": labels,
    "class_names": class_names,
    "class_count": dict(zip(*np.unique(y, return_counts=True))),
    "pct_typical": 100 * np.mean(y == 0),
    "pct_popular": 100 * np.mean(y == 1),
    "pct_viral": 100 * np.mean(y == 2),
}
joblib.dump(meta, "model_meta.pkl")
print(" Saved: best_recipe_popularity_model.pkl, model_meta.pkl")

# Plot confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names, cmap='Blues')
plt.title("Confusion Matrix: Popularity Classes")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print(" Graph saved: confusion_matrix.png")
