"""
Re-trains the tutor subject classifier and saves tutor_model.pkl.
This mirrors the pipeline in code.ipynb exactly, so the saved bundle
matches whatever scikit-learn version is currently installed.

Run once with:  python train_model.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("tutors_ml_dataset.csv")

# --- Encode categorical columns (group rare countries into 'Other') ---
country_freq = df["country"].value_counts()
df["country_grouped"] = df["country"].apply(lambda c: c if country_freq[c] >= 5 else "Other")
cat_dummies = pd.get_dummies(df[["country_grouped", "price_unit"]], prefix=["country", "unit"])

# --- Combine text columns ---
df["all_text"] = df["subjects_taught"].fillna("") + " " + df["description"].fillna("")

# --- Encode target ---
le = LabelEncoder()
y = le.fit_transform(df["search_subject"])

# --- Numeric features ---
num_cols = ["total_experience", "online_experience", "price_min_usd",
            "price_max_usd", "price_avg_usd", "num_subjects", "desc_length"]
base_features = pd.concat([df[num_cols], cat_dummies], axis=1).reset_index(drop=True)

# --- Stratified train/test split ---
all_index = np.arange(len(df))
train_idx, test_idx = train_test_split(all_index, test_size=0.20, random_state=42, stratify=y)

# --- Cap price outliers at 99th percentile (train-only thresholds) ---
price_cols = ["price_min_usd", "price_max_usd", "price_avg_usd"]
for col in price_cols:
    cap = df[col].iloc[train_idx].quantile(0.99)
    df[col] = df[col].clip(upper=cap)
base_features = pd.concat([df[num_cols], cat_dummies], axis=1).reset_index(drop=True)

# --- TF-IDF (fit on train only) ---
tfidf = TfidfVectorizer(max_features=250, stop_words="english")
text_train = tfidf.fit_transform(df["all_text"].iloc[train_idx]).toarray()
text_test = tfidf.transform(df["all_text"].iloc[test_idx]).toarray()

# --- Scale base features (fit on train only) ---
scaler = StandardScaler()
base_train = scaler.fit_transform(base_features.iloc[train_idx])
base_test = scaler.transform(base_features.iloc[test_idx])

X_train = np.hstack([base_train, text_train])
X_test = np.hstack([base_test, text_test])
y_train, y_test = y[train_idx], y[test_idx]

# --- Train the same three models and pick the best ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name:22s} -> accuracy = {acc:.3f}")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name} (accuracy = {results[best_name]:.3f})")

# --- Save the bundle ---
bundle = {
    "model": models[best_name],
    "scaler": scaler,
    "tfidf": tfidf,
    "le": le,
    "num_cols": num_cols,
    "cat_columns": cat_dummies.columns.tolist(),
}
joblib.dump(bundle, "tutor_model.pkl")
print("Saved tutor_model.pkl")
