# Phishing Email Detection System

A Machine Learning-based web application that detects phishing emails using Natural Language Processing (NLP) and classification algorithms. Built as a **BCA Cyber Security Mini Project**.

## Project Overview

Phishing is a cyber-attack where attackers send fraudulent emails disguised as legitimate sources to steal sensitive information such as passwords, credit card numbers, and personal data. This project builds an intelligent web application that automatically classifies emails as **Phishing** or **Safe** using three ML models and automatically selects the best-performing one.

### What It Does

- Classifies pasted email text as **Phishing** or **Safe**
- Displays a **confidence score** and **risk level** (Low, Medium, High)
- **Highlights suspicious URLs** and **phishing keywords** found in the email
- Compares three ML models (Logistic Regression, Naive Bayes, Random Forest) and auto-selects the best
- Shows full **model evaluation metrics** (accuracy, precision, recall, F1, confusion matrix, classification report)
- Tracks **prediction history** within the session

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Flask 3.0 | Web framework |
| Scikit-learn 1.5 | Machine learning (models, TF-IDF, metrics) |
| NLTK 3.8 | Natural language processing (tokenization, stop words) |
| Pandas 2.2 | Data manipulation |
| Joblib 1.4 | Model serialization |
| Bootstrap 5.3 | Responsive UI framework |
| Inter Font | Modern typography |

---

## Folder Structure

```
phishing-email-detection/
├── app.py                  # Flask web application (routes, templates)
├── train_model.py          # Training pipeline (preprocess, TF-IDF, train, evaluate, save)
├── predict.py              # Prediction module (load model, predict, extract indicators)
├── generate_dataset.py     # Script to generate the phishing email dataset
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── dataset/
│   ├── phishing_emails.csv # Labeled dataset (email_text, label)
│   └── .gitkeep
├── models/
│   ├── best_model.joblib   # Saved best ML model
│   ├── tfidf_vectorizer.joblib  # Saved TF-IDF vectorizer
│   ├── metrics.json        # Evaluation metrics for web UI
│   └── .gitkeep
├── templates/
│   ├── base.html           # Base template (navbar, footer)
│   ├── index.html          # Home page
│   ├── detect.html         # Email detection page
│   ├── about.html          # About page
│   ├── history.html        # Prediction history page
│   ├── metrics.html        # Model evaluation page
│   └── error.html          # Error page (404, 500)
└── static/
    ├── css/
    │   └── style.css        # Custom styles
    └── js/
        └── main.js          # Custom JavaScript
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone or Download

```bash
git clone <your-repository-url>
cd phishing-email-detection
```

### Step 2: Create a Virtual Environment (recommended)

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

NLTK resources are downloaded automatically on first run. If you encounter issues, run:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Commands to Run

### 1. Generate the Dataset

```bash
python generate_dataset.py
```

This creates `dataset/phishing_emails.csv` with 240 labeled emails (120 phishing, 120 safe).

### 2. Train the Models

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Extract TF-IDF features
- Train Logistic Regression, Naive Bayes, and Random Forest
- Evaluate and compare all three models
- Save the best model and vectorizer to `models/`
- Write evaluation metrics to `models/metrics.json`

### 3. Start the Web Application

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## Using the Application

### Home Page (`/`)
Project overview with statistics and feature highlights.

### Detect Page (`/detect`)
1. Paste the full email content into the text box
2. Click **Detect Phishing**
3. View the results:
   - **Prediction**: Phishing or Safe
   - **Confidence Score**: Percentage confidence
   - **Risk Level**: Low, Medium, or High
   - **Suspicious URLs**: List of URLs found
   - **Phishing Keywords**: List of suspicious phrases
   - **Highlighted Email**: Email text with URLs and keywords highlighted

### Metrics Page (`/metrics`)
View model comparison, accuracy visualization, confusion matrices, and classification reports.

### History Page (`/history`)
Review past predictions from the current session.

### About Page (`/about`)
Detailed project description, methodology, and technology stack.

---

## API Endpoint

The application also exposes a JSON API for programmatic access:

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Dear customer, your account has been suspended. Verify at http://secure-login-verify.com/login"}'
```

**Response:**
```json
{
  "prediction": "Phishing",
  "label": 1,
  "confidence": 95.5,
  "risk_level": "High",
  "urls_found": ["http://secure-login-verify.com/login"],
  "keywords_found": ["account suspended", "verify your account"],
  "suspicious_count": 2,
  "model_name": "Logistic Regression"
}
```

---

## Machine Learning Pipeline

### 1. Text Preprocessing
- Convert to lowercase
- Replace URLs with a placeholder token
- Remove punctuation
- Remove digits
- Tokenize using NLTK `word_tokenize`
- Remove English stop words
- Filter tokens shorter than 3 characters

### 2. Feature Extraction
- **TF-IDF Vectorizer** with up to 5,000 features
- Unigram and bigram ranges (1, 2)
- Converts text into a numeric feature matrix

### 3. Models Trained
| Model | Key Parameter |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Multinomial Naive Bayes | default |
| Random Forest | `n_estimators=100` |

### 4. Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: 2x2 matrix of predictions vs. actual
- **Classification Report**: Per-class precision, recall, F1, support

### 5. Model Selection
The model with the highest **F1 score** is automatically selected and saved.

---

## Sample Screenshots

> Add your screenshots here after running the application.

1. **Home Page** — Hero section with project overview and statistics
2. **Detect Page** — Email input form with prediction results
3. **Phishing Detection Result** — Red alert with highlighted URLs and keywords
4. **Safe Detection Result** — Green alert with confidence score
5. **Metrics Page** — Model comparison table, accuracy bars, and confusion matrices
6. **History Page** — Table of past predictions
7. **About Page** — Project description and technology stack

---

## License

This project is created for academic purposes as part of the BCA Cyber Security curriculum.

## Author

**Student Name** — BCA Cyber Security Mini Project

---

> **Disclaimer**: This tool is for educational purposes only and should not be used as the sole method for phishing detection in production environments.
