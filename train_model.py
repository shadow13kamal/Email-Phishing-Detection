"""Train and evaluate phishing detection models.

Pipeline:
    1. Load the dataset from dataset/phishing_emails.csv
    2. Preprocess email text (lowercase, strip punctuation, remove stop
       words, tokenize).
    3. Convert text to numeric features with TF-IDF.
    4. Train three classifiers — Logistic Regression, Naive Bayes,
       and Random Forest.
    5. Evaluate each model (accuracy, precision, recall, F1, confusion
       matrix, classification report).
    6. Pick the best model by F1 score, save it with Joblib, and write
       evaluation metrics to models/metrics.json for the web UI.

Usage:
    python train_model.py
"""

import json
import os
import re
import string

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("dataset", "phishing_emails.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Ensure NLTK stop-word and tokenizer resources are available
for resource in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{resource}" if "stop" in resource else f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Lowercase, remove punctuation/stop words, and tokenize email text."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Replace URLs with a placeholder token so the model learns the pattern
    text = re.sub(r"https?://\S+|www\.\S+", " urlplaceholder ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Tokenize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        tokens = word_tokenize(text)

    # Remove stop words and short tokens
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def load_data():
    """Load the CSV dataset and return raw texts, labels."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run 'python generate_dataset.py' first."
        )

    df = pd.read_csv(DATASET_PATH)
    if "email_text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Dataset must contain 'email_text' and 'label' columns."
        )

    df = df.dropna(subset=["email_text", "label"])
    df["label"] = df["label"].astype(int)
    return df["email_text"].values, df["label"].values


def evaluate_model(name, model, X_test, y_test):
    """Return a dict of metrics for a trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "name": name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_and_save():
    """Run the full training pipeline and persist the best model."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  PHISHING EMAIL DETECTION — MODEL TRAINING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading dataset...")
    texts, labels = load_data()
    print(f"      Loaded {len(texts)} emails "
          f"({sum(labels)} phishing, {len(labels) - sum(labels)} safe)")

    # 2. Preprocess
    print("\n[2/6] Preprocessing text (lowercase, punctuation, stop words)...")
    processed_texts = [preprocess_text(t) for t in texts]

    # 3. TF-IDF feature extraction
    print("\n[3/6] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed_texts)
    y = labels
    print(f"      Feature matrix shape: {X.shape}")

    # 4. Train-test split
    print("\n[4/6] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # 5. Train three models
    print("\n[5/6] Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    all_metrics = {}
    trained_models = {}

    for name, model in models.items():
        print(f"      → Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        trained_models[name] = model
        all_metrics[name] = metrics
        print(f"        Accuracy: {metrics['accuracy']:.4f}  "
              f"F1: {metrics['f1_score']:.4f}")

    # 6. Compare and save the best model
    print("\n[6/6] Comparing models and saving the best...")
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1_score"])
    best_model = trained_models[best_name]

    print(f"\n      Best model: {best_name} "
          f"(F1 = {all_metrics[best_name]['f1_score']:.4f})")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"      Saved model → {MODEL_PATH}")
    print(f"      Saved vectorizer → {VECTORIZER_PATH}")

    # Write metrics JSON for the web UI
    metrics_output = {
        "best_model": best_name,
        "models": all_metrics,
        "dataset_size": len(texts),
        "feature_count": X.shape[1],
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"      Saved metrics → {METRICS_PATH}")

    # Print summary table
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} "
          f"{'Recall':<12} {'F1 Score':<12}")
    print("-" * 73)
    for name, m in all_metrics.items():
        marker = " ★" if name == best_name else ""
        print(f"{name + marker:<25} {m['accuracy']:<12.4f} "
              f"{m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1_score']:<12.4f}")
    print("=" * 60)

    # Print confusion matrix for the best model
    print(f"\nConfusion Matrix ({best_name}):")
    cm = all_metrics[best_name]["confusion_matrix"]
    print(f"  True Safe     → Pred Safe: {cm[0][0]}  Pred Phishing: {cm[0][1]}")
    print(f"  True Phishing → Pred Safe: {cm[1][0]}  Pred Phishing: {cm[1][1]}")

    print("\nTraining complete!\n")


if __name__ == "__main__":
    train_and_save()
