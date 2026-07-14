"""Vercel serverless entry point.

Vercel looks for Python files in the `api/` directory and treats each
as a serverless function. This file imports the Flask app from the
parent directory and exposes it as a WSGI application.
"""

import sys
import os

# Add the project root to sys.path so `app` and `predict` can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app

# Vercel expects a variable named `app` that is a WSGI callable
