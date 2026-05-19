import os
import base64
import sqlite3
from pathlib import Path
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify,
)
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")

MODEL_PATH = Path("best.pt")
if not MODEL_PATH.exists():
    raise FileNotFoundError("best.pt not found. Place model weights in the project root.")
model = YOLO(str(MODEL_PATH))

# "protectuve boots" is a typo in the original dataset labels
REQUIRED_PPE = {"protective helmet", "jacket", "glove", "protectuve boots"}

DB = "database.db"


def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def run_detection(image_bgr):
    results = model.predict(image_bgr, conf=0.5, verbose=False)
    detected = {results[0].names[int(b.cls)].lower() for b in results[0].boxes}
    missing = REQUIRED_PPE - detected
    annotated = results[0].plot()
    return annotated, detected, missing


def encode_b64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not all([username, email, password]):
            flash("All fields are required.", "error")
            return render_template("register.html")

        try:
            with sqlite3.connect(DB) as conn:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, generate_password_hash(password)),
                )
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already in use.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        with sqlite3.connect(DB) as conn:
            conn.row_factory = sqlite3.Row
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/detect", methods=["POST"])
@login_required
def detect():
    if "image" not in request.files or not request.files["image"].filename:
        flash("Please select an image file.", "error")
        return redirect(url_for("dashboard"))

    file = request.files["image"]
    if Path(file.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        flash("Unsupported file type. Use JPG, PNG, or WEBP.", "error")
        return redirect(url_for("dashboard"))

    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        flash("Could not read the image.", "error")
        return redirect(url_for("dashboard"))

    annotated, detected, missing = run_detection(img)

    return render_template(
        "result.html",
        original=encode_b64(img),
        annotated=encode_b64(annotated),
        detected=sorted(detected),
        missing=sorted(missing),
    )


@app.route("/detect_frame", methods=["POST"])
@login_required
def detect_frame():
    raw = request.json.get("frame", "")
    if "," in raw:
        raw = raw.split(",", 1)[1]

    img_array = np.frombuffer(base64.b64decode(raw), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    annotated, detected, missing = run_detection(frame)

    return jsonify({
        "annotated": "data:image/jpeg;base64," + encode_b64(annotated),
        "detected": sorted(detected),
        "missing": sorted(missing),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


init_db()

if __name__ == "__main__":
    app.run(debug=True)
