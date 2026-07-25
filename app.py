import os
import base64
import sqlite3
import threading
from pathlib import Path
from functools import wraps
from uuid import uuid4

import cv2
import numpy as np
import imageio.v2 as imageio
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify,
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from detector import PPEDetector

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB upload cap

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PROCESSED_DIR = Path("static/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job tracking for background video processing.
# Fine for a single-process student deployment; wouldn't survive a
# multi-worker/multi-process server without moving this to shared storage.
VIDEO_JOBS = {}
VIDEO_JOBS_LOCK = threading.Lock()

MODEL_PATH = Path("models/best.onnx")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "models/best.onnx not found. Run: python export_onnx.py"
    )
model = PPEDetector(MODEL_PATH)

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


def run_detection(image_bgr, conf=None):
    annotated, detected = model.predict(image_bgr, conf=conf)
    missing = REQUIRED_PPE - detected
    return annotated, detected, missing


def parse_conf(raw, default=0.3):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return min(max(value, 0.05), 0.95)


def set_job(job_id, **fields):
    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id].update(fields)


def process_video_job(job_id, input_path, conf):
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        # Not always accurate for every container, but good enough for a progress estimate.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        output_name = f"{job_id}.mp4"
        output_path = PROCESSED_DIR / output_name
        # libx264 (not cv2.VideoWriter's mp4v) so the result actually plays inline in browsers.
        writer = imageio.get_writer(
            str(output_path), fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=None,
        )

        frame_count = 0
        compliant_count = 0
        ever_detected, ever_missing = set(), set()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            annotated, detected, missing = run_detection(frame, conf=conf)
            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            frame_count += 1
            ever_detected |= detected
            if missing:
                ever_missing |= missing
            else:
                compliant_count += 1

            if total_frames:
                progress = min(99, round(100 * frame_count / total_frames))
                set_job(job_id, progress=progress)

        cap.release()
        writer.close()

        if frame_count == 0:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("No frames could be read from that video.")

        set_job(
            job_id,
            status="done",
            progress=100,
            output_name=output_name,
            frame_count=frame_count,
            compliant_pct=round(100 * compliant_count / frame_count),
            ever_detected=sorted(ever_detected),
            ever_missing=sorted(ever_missing),
        )
    except Exception as e:
        set_job(job_id, status="error", error=str(e))
    finally:
        Path(input_path).unlink(missing_ok=True)


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

    conf = parse_conf(request.form.get("conf"))
    annotated, detected, missing = run_detection(img, conf=conf)

    return render_template(
        "result.html",
        original=encode_b64(img),
        annotated=encode_b64(annotated),
        detected=sorted(detected),
        missing=sorted(missing),
    )


@app.route("/detect_video", methods=["POST"])
@login_required
def detect_video():
    if "video" not in request.files or not request.files["video"].filename:
        return jsonify({"error": "Please select a video file."}), 400

    file = request.files["video"]
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": "Unsupported file type. Use MP4, AVI, MOV, MKV, or WEBM."}), 400

    conf = parse_conf(request.form.get("conf"))

    job_id = uuid4().hex
    upload_path = PROCESSED_DIR / f"{job_id}_in{suffix}"
    file.save(upload_path)

    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {"status": "processing", "progress": 0}

    threading.Thread(
        target=process_video_job, args=(job_id, upload_path, conf), daemon=True
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/video_status/<job_id>")
@login_required
def video_status(job_id):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/video_result/<job_id>")
@login_required
def video_result(job_id):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job or job.get("status") != "done":
        flash("That video result isn't ready or has expired.", "error")
        return redirect(url_for("dashboard"))

    return render_template(
        "result_video.html",
        video_url=url_for("static", filename=f"processed/{job['output_name']}"),
        frame_count=job["frame_count"],
        compliant_pct=job["compliant_pct"],
        ever_detected=job["ever_detected"],
        ever_missing=job["ever_missing"],
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

    conf = parse_conf(request.json.get("conf"))
    annotated, detected, missing = run_detection(frame, conf=conf)

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
    app.run(debug=True, threaded=True)
