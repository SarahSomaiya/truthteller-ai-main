"""
api/index.py
────────────
Vercel serverless entry point for TruthTeller AI.

Vercel auto-discovers Python files inside the `api/` directory
and exposes them as serverless functions. This file imports and
re-exports the Flask `app` so Vercel can call it via WSGI.
"""

import sys
import os

# Make backend/ importable from this file's parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app  # noqa: F401  — Vercel needs the `app` name

# Vercel looks for a callable named `app` (WSGI) in this module.
