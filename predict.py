"""Prediction module for phishing email detection.

Loads the saved TF-IDF vectorizer and best model, preprocesses incoming
email text, predicts phishing/safe, and extracts suspicious indicators
(URLs and phishing keywords) for highlighting in the UI.

Used by app.py — import `PhishingDetector` and call `.predict(email_text)`.
"""

import json
import os
import re
import string

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------
for resource in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(
            f"corpora/{resource}" if "stop" in resource else f"tokenizers/{resource}"
        )
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
# Suspicious keywords (lowercase) for indicator detection
# ---------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "verify your account",
    "confirm your identity",
    "account suspended",
    "account locked",
    "account will be closed",
    "urgent",
    "immediate action",
    "click here",
    "click below",
    "claim your prize",
    "you have won",
    "lottery",
    "congratulations you",
    "limited time",
    "act now",
    "update your information",
    "verify your billing",
    "reset your password",
    "security alert",
    "unusual activity",
    "enter your password",
    "social security number",
    "bank details",
    "credit card",
    "wire transfer",
    "tax refund",
    "irs",
    "bitcoin",
    "gift card",
    "no credit check",
    "free",
    "winner",
    "selected for",
    "log in with your email",
    "shipping address",
    "pay a small fee",
    "legal action",
    "unpaid invoice",
    "auto-renewed",
    "infected with",
    "download the security tool",
    "wallet passphrase",
    "date of birth",
]

# Regex to find URLs (http, https, www, and bare-domain patterns)
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+|[a-zA-Z0-9.-]+\.(?:com|net|org|tk|ml|cf|ga|gq|io|co|info|biz)[^\s<>\"']*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Preprocessing (must match train_model.py exactly)
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Lowercase, replace URLs, remove punctuation/digits/stop words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urlplaceholder ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Risk-level helper
# ---------------------------------------------------------------------------
def calculate_risk_level(confidence, suspicious_count):
    """Return a risk-level string based on confidence and indicator count."""
    if confidence >= 0.85 and suspicious_count >= 3:
        return "High"
    if confidence >= 0.7 or suspicious_count >= 2:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------
class PhishingDetector:
    """Load the trained model and run predictions on email text."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python train_model.py' first."
            )
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Vectorizer not found at {VECTORIZER_PATH}. "
                "Run 'python train_model.py' first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _extract_urls(self, text):
        """Return a list of URLs found in the raw email text."""
        urls = URL_PATTERN.findall(text)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for url in urls:
            url = url.rstrip(".,;:!?")
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    def _extract_keywords(self, text):
        """Return a list of suspicious keyword phrases found in the email."""
        lower_text = text.lower()
        found = []
        for keyword in SUSPICIOUS_KEYWORDS:
            if keyword in lower_text:
                found.append(keyword)
        return found

    def _highlight(self, text, urls, keywords):
        """Return HTML with URLs and keywords highlighted via <mark> spans."""
        highlighted = text

        # Escape HTML special characters first
        highlighted = (
            highlighted.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        # Highlight URLs
        for url in urls:
            escaped_url = (
                url.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            if escaped_url and escaped_url in highlighted:
                highlighted = highlighted.replace(
                    escaped_url,
                    f'<mark class="highlight-url">{escaped_url}</mark>',
                )

        # Highlight keywords (case-insensitive)
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted = pattern.sub(
                r'<mark class="highlight-keyword">\g<0></mark>',
                highlighted,
            )

        return highlighted

    def predict(self, email_text):
        """Run the full prediction pipeline on a single email.

        Returns a dict with:
            prediction, confidence, risk_level, urls, keywords,
            highlighted_html, model_name
        """
        if not email_text or not email_text.strip():
            raise ValueError("Email text cannot be empty.")

        # Preprocess and vectorize
        processed = preprocess_text(email_text)
        features = self.vectorizer.transform([processed])

        # Predict
        prediction_label = int(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(max(probabilities))

        # Extract indicators
        urls = self._extract_urls(email_text)
        keywords = self._extract_keywords(email_text)
        suspicious_count = len(urls) + len(keywords)

        # Risk level
        risk_level = calculate_risk_level(confidence, suspicious_count)

        # Highlighted HTML
        highlighted_html = self._highlight(email_text, urls, keywords)

        result = {
            "prediction": "Phishing" if prediction_label == 1 else "Safe",
            "label": prediction_label,
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level,
            "urls_found": urls,
            "keywords_found": keywords,
            "suspicious_count": suspicious_count,
            "highlighted_html": highlighted_html,
            "model_name": self.metrics.get("best_model", "Unknown"),
        }
        return result

    def get_metrics(self):
        """Return the evaluation metrics dict for display."""
        return self.metrics
