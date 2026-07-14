"""Flask web application for Phishing Email Detection.

Routes:
    /        — Home page with project overview
    /detect  — Email detection page (paste email → get prediction)
    /about   — About page with project details
    /history — Prediction history page (stored in session)
    /metrics — Model evaluation metrics page

Run:  python app.py  →  http://127.0.0.1:5000
"""

import os
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, jsonify

from predict import PhishingDetector

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "phishing-detector-secret-key-2024")

# Store up to 50 predictions in session
MAX_HISTORY = 50

# Initialise the detector once at startup
detector = None
try:
    detector = PhishingDetector()
    print(f"[INFO] Model loaded: {detector.metrics.get('best_model', 'Unknown')}")
except FileNotFoundError as exc:
    print(f"[WARNING] {exc}")
    print("[WARNING] Run 'python train_model.py' before starting the app.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Home page with project overview and quick stats."""
    metrics = detector.get_metrics() if detector else {}
    model_name = metrics.get("best_model", "Not trained")
    dataset_size = metrics.get("dataset_size", 0)
    return render_template(
        "index.html",
        model_name=model_name,
        dataset_size=dataset_size,
        metrics=metrics,
    )


@app.route("/detect", methods=["GET", "POST"])
def detect():
    """Email detection page — accepts pasted email text and returns a prediction."""
    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if not email_text:
            return render_template(
                "detect.html",
                error="Please paste an email to analyze.",
                result=None,
            )

        if detector is None:
            return render_template(
                "detect.html",
                error="Model is not loaded. Run 'python train_model.py' first.",
                result=None,
            )

        try:
            result = detector.predict(email_text)

            # Save to session history
            history = session.get("history", [])
            history.insert(0, {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "risk_level": result["risk_level"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "email_preview": email_text[:150] + ("..." if len(email_text) > 150 else ""),
            })
            session["history"] = history[:MAX_HISTORY]

            return render_template("detect.html", result=result, error=None)

        except Exception as exc:
            return render_template(
                "detect.html",
                error=f"An error occurred during analysis: {exc}",
                result=None,
            )

    return render_template("detect.html", result=None, error=None)


@app.route("/about")
def about():
    """About page with project description and technology details."""
    return render_template("about.html")


@app.route("/history")
def history():
    """Prediction history page (stored in session)."""
    history = session.get("history", [])
    return render_template("history.html", history=history)


@app.route("/history/clear")
def clear_history():
    """Clear the prediction history stored in session."""
    session.pop("history", None)
    return redirect(url_for("history"))


@app.route("/metrics")
def metrics():
    """Model evaluation metrics page."""
    metrics_data = detector.get_metrics() if detector else {}
    return render_template("metrics.html", metrics=metrics_data)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """JSON API endpoint for programmatic access."""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "email_text" not in data:
        return jsonify({"error": "Missing 'email_text' field"}), 400

    email_text = data["email_text"].strip()
    if not email_text:
        return jsonify({"error": "Email text cannot be empty"}), 400

    try:
        result = detector.predict(email_text)
        # Remove HTML from API response
        result.pop("highlighted_html", None)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(exc):
    return render_template("error.html", code=404,
                           message="Page not found"), 404


@app.errorhandler(500)
def internal_error(exc):
    return render_template("error.html", code=500,
                           message="Internal server error"), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
