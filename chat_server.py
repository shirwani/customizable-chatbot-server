from __future__ import annotations
import sys
from flask import Flask, jsonify, request
sys.path.append('')  # Add the code directory to the Python path for imports
from utils import *
from execute_prompt import do_execute_prompt

app = Flask(__name__)

# Enable cross-origin calls from the local http server / embedded sites.
# This avoids browser "TypeError: Failed to fetch" caused by CORS blocking.
allowed_origins = {
    "http://127.0.0.1:8001",
    "http://localhost:8001",
}

try:
    from flask_cors import CORS  # type: ignore

    CORS(
        app,
        origins=list(allowed_origins),
        supports_credentials=False,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=86400,
        automatic_options=True,
    )
except Exception:
    # If flask-cors isn't installed yet, we'll still add headers manually below.
    pass


@app.after_request
def add_cors_headers(response):
    """Ensure all responses include suitable CORS headers.

    This covers cases where flask-cors is not installed or not applied.
    """
    origin = request.headers.get("Origin", "")
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"
    return response


# RAG server


@app.route("/", methods=["GET", "POST", "OPTIONS"])
def home():
    # Handle CORS preflight explicitly just in case
    if request.method == "OPTIONS":
        resp = jsonify({"ok": True})
        return resp, 200

    # For GET requests from the widget, the question usually comes in as a
    # query parameter. For POST, it could be JSON or form-encoded; handle both.
    if request.method == "GET":
        prompt = request.args.get("prompt", "What are your hours?")
    else:
        json_data = request.get_json(silent=True) or {}
        prompt = (
            json_data.get("prompt")
            or request.form.get("prompt")
            or request.args.get("prompt", "What are your hours?")
        )

    print("Received prompt:", prompt)

    answer = do_execute_prompt(prompt)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    # Bind on all interfaces so the client (running on a different port) can reach it
    app.run(host="0.0.0.0", port=8001, debug=True)
