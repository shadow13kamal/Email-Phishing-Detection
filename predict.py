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

# ---------------------------------------------------------------------------
# Paths (absolute so they work on Vercel and locally)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Stop words (hardcoded fallback so NLTK download is not required on Vercel)
# ---------------------------------------------------------------------------
_FALLBACK_STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am',
    'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because',
    'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',
    'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does',
    'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn',
    "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll",
    "he's", 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is',
    'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'just',
    'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn',
    "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not',
    'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan',
    "shan't", 'she', "she'd", "she'll", "she's", 'should', "should've",
    'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that',
    "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', "they'd", "they'll", "they're", "they've",
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've',
    'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're",
    "we've", 'were', 'weren', "weren't", 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't",
    'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves',
}

STOP_WORDS = set(_FALLBACK_STOP_WORDS)

# Try to use NLTK stopwords if available (for local dev consistency)
try:
    from nltk.corpus import stopwords as _nltk_stopwords
    STOP_WORDS = set(_nltk_stopwords.words("english"))
except Exception:
    pass


def _tokenize(text):
    """Simple regex tokenizer — no NLTK dependency needed."""
    return re.findall(r"\b\w\w+\b", text)


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
    tokens = _tokenize(text)
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
        highlighted = (
            highlighted.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
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
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted = pattern.sub(
                r'<mark class="highlight-keyword">\g<0></mark>',
                highlighted,
            )
        return highlighted

    def predict(self, email_text):
        """Run the full prediction pipeline on a single email."""
        if not email_text or not email_text.strip():
            raise ValueError("Email text cannot be empty.")

        processed = preprocess_text(email_text)
        features = self.vectorizer.transform([processed])

        prediction_label = int(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(max(probabilities))

        urls = self._extract_urls(email_text)
        keywords = self._extract_keywords(email_text)
        suspicious_count = len(urls) + len(keywords)

        risk_level = calculate_risk_level(confidence, suspicious_count)
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
