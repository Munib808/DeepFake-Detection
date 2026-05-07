"""
╔══════════════════════════════════════════════════════════════════════╗
║           DEEPFAKE DETECTION SYSTEM — Flask Web Application         ║
║                                                                      ║
║  Model: EfficientNetB0 + Bidirectional LSTM (dual-input)            ║
║  Face Detection: InsightFace SCRFD + 106-pt landmarks               ║
║  Features: 14-dim biometric vector per frame                        ║
╚══════════pi════════════════════════════════════════════════════════════╝
"""

import os
import uuid
import math
import time
import logging
import warnings
import shutil

import cv2
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, redirect, url_for, jsonify

# ═══════════════════════════════════════════════════════════════════════
# App Configuration
# ═══════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["FRAMES_FOLDER"] = os.path.join("static", "extracted_frames")
app.config["FACES_FOLDER"] = os.path.join("static", "extracted_faces")

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Create folders
for folder in [app.config["UPLOAD_FOLDER"],
               app.config["FRAMES_FOLDER"],
               app.config["FACES_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Model Configuration (matches your training notebook exactly)
# ═══════════════════════════════════════════════════════════════════════

class CFG:
    SHORT_THRESHOLD = 5       # seconds
    LONG_THRESHOLD  = 32      # seconds
    MAX_FRAMES      = 32
    MIN_FRAMES      = 2
    N_FRAMES        = 32      # model expects exactly 32 timesteps
    IMG_SIZE        = 224
    FEATURE_COUNT   = 14      # 14-dim biometric feature vector
    RESIZE_BATCH    = 32
    EAR_BLINK_THRESH = 0.21
    CONSEC_FRAMES    = 2


# ═══════════════════════════════════════════════════════════════════════
# InsightFace 106-Point Landmark Indices (from your notebook)
# ═══════════════════════════════════════════════════════════════════════

RIGHT_EYE_106       = [33, 34, 35, 37, 40, 41]
LEFT_EYE_106        = [93, 92, 91, 87, 88, 95]
MOUTH_106           = [52, 61, 55, 67, 57, 65, 59, 63]
RIGHT_EYE_OUTER_106 = 33
LEFT_EYE_OUTER_106  = 93
MOUTH_RIGHT_106     = 52
MOUTH_LEFT_106      = 61
NOSE_TIP_106        = 86
CHIN_106            = 16


# ═══════════════════════════════════════════════════════════════════════
# Biometric Calculators (identical to your training code)
# ═══════════════════════════════════════════════════════════════════════

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_EAR(eye_pts):
    """Eye Aspect Ratio — identical formula to training."""
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_MAR(mouth_pts):
    """Mouth Aspect Ratio — identical formula to training."""
    vert1 = euclidean(mouth_pts[2], mouth_pts[6])
    vert2 = euclidean(mouth_pts[3], mouth_pts[5])
    horiz = euclidean(mouth_pts[0], mouth_pts[1])
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)


def compute_symmetry(lms_dict):
    """Facial symmetry — identical formula to training."""
    nose    = lms_dict[NOSE_TIP_106]
    left_d  = euclidean(lms_dict[RIGHT_EYE_OUTER_106], nose)
    right_d = euclidean(lms_dict[LEFT_EYE_OUTER_106], nose)
    return 1.0 - abs(left_d - right_d) / (left_d + right_d + 1e-6)


# ═══════════════════════════════════════════════════════════════════════
# GPU Helpers (adapted for CPU/GPU inference)
# ═══════════════════════════════════════════════════════════════════════

def gpu_resize_batch(images, target_h, target_w):
    """Batch-resize face crops."""
    resized = tf.image.resize(images, [target_h, target_w],
                              method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(resized, tf.uint8)


def gpu_image_quality_batch(face_crops_uint8):
    """
    Batched brightness, contrast, laplacian variance.
    Returns: (N, 3) float32 [brightness, contrast, lap_var]
    """
    faces_f = tf.cast(face_crops_uint8, tf.float32)
    gray = tf.reduce_sum(
        faces_f * tf.constant([0.2989, 0.5870, 0.1140]), axis=-1)
    brightness = tf.reduce_mean(gray, axis=[1, 2]) / 255.0
    contrast   = tf.math.reduce_std(gray, axis=[1, 2]) / 128.0

    lap_kernel = tf.constant(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
    lap_kernel = tf.reshape(lap_kernel, [3, 3, 1, 1])
    gray_4d    = tf.reshape(gray, [-1, CFG.IMG_SIZE, CFG.IMG_SIZE, 1])
    laplacian  = tf.nn.conv2d(gray_4d, lap_kernel,
                              strides=[1, 1, 1, 1], padding='SAME')
    lap_mean   = tf.reduce_mean(laplacian, axis=[1, 2, 3])
    lap_sq     = tf.reduce_mean(tf.square(laplacian), axis=[1, 2, 3])
    lap_var    = (lap_sq - tf.square(lap_mean)) / 1000.0
    return tf.stack([brightness, contrast, lap_var], axis=1)


# ═══════════════════════════════════════════════════════════════════════
# Load InsightFace + Keras Model at Startup
# ═══════════════════════════════════════════════════════════════════════

logger.info("Loading InsightFace models...")
try:
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(
        allowed_modules=['detection', 'landmark_2d_106'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("✓ InsightFace ready.")
except Exception as e:
    logger.warning(f"InsightFace GPU failed, trying CPU: {e}")
    face_app = FaceAnalysis(
        allowed_modules=['detection', 'landmark_2d_106'],
        providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    logger.info("✓ InsightFace ready (CPU).")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_best_model.keras")
logger.info(f"Loading Keras model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
logger.info("✓ Keras model loaded.")


# ═══════════════════════════════════════════════════════════════════════
# Video Processing Pipeline (adapted from your VideoProcessor class)
# ═══════════════════════════════════════════════════════════════════════

def detect_face(frame_rgb):
    """Run InsightFace: SCRFD detection + 106 landmarks."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    faces = face_app.get(frame_bgr)
    if not faces:
        return None, None
    face = faces[0]
    lmk_106 = face.landmark_2d_106
    if lmk_106 is None:
        return None, None
    lms_dict = {i: (int(lmk_106[i][0]), int(lmk_106[i][1]))
                for i in range(106)}
    bbox = face.bbox.astype(int)
    return bbox, lms_dict


def get_frame_indices(total_frames, fps, duration_sec):
    """Adaptive frame sampling — identical to training."""
    if duration_sec <= CFG.SHORT_THRESHOLD:
        n       = min(total_frames, CFG.MAX_FRAMES)
        indices = list(np.linspace(0, total_frames - 1, n, dtype=int))
    elif duration_sec <= CFG.LONG_THRESHOLD:
        step    = max(1, int(fps))
        indices = list(range(0, total_frames, step))
        if len(indices) > CFG.MAX_FRAMES:
            indices = [indices[i] for i in
                       np.linspace(0, len(indices) - 1,
                                   CFG.MAX_FRAMES, dtype=int)]
    else:
        indices = list(np.linspace(
            0, total_frames - 1, CFG.MAX_FRAMES, dtype=int))
    return indices


def process_video(video_path, session_id):
    """
    Full pipeline: extract frames → detect faces → compute features → predict.
    Returns a result dict for the template.
    """
    frames_dir = os.path.join(app.config["FRAMES_FOLDER"], session_id)
    faces_dir  = os.path.join(app.config["FACES_FOLDER"], session_id)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    # ── Read video metadata ───────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_indices = get_frame_indices(total_frames, fps, duration_sec)
    n_to_sample   = len(frame_indices)

    if n_to_sample < CFG.MIN_FRAMES:
        return {"error": "Video too short — not enough frames to analyze."}

    # ── Batch-read frames ─────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    target_set  = set(frame_indices)
    max_target  = max(frame_indices)
    frames_dict = {}
    fi = 0
    while fi <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in target_set:
            frames_dict[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fi += 1
    cap.release()

    indexed_frames = [(idx, frames_dict[idx])
                      for idx in frame_indices if idx in frames_dict]

    if len(indexed_frames) < CFG.MIN_FRAMES:
        return {"error": "Could not read enough frames from video."}

    # ── Face detection + feature extraction ───────────────────────────
    raw_ear       = []
    raw_iod       = []
    per_frame     = []
    raw_crops     = []
    frame_paths   = []
    face_paths    = []
    frame_details = []

    for seq_idx, (fi, rgb) in enumerate(indexed_frames):
        h, w = rgb.shape[:2]

        # Save extracted frame
        frame_filename = f"frame_{seq_idx:03d}.jpg"
        frame_save_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_save_path,
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frame_web_path = f"extracted_frames/{session_id}/{frame_filename}"

        bbox, lms_dict = detect_face(rgb)

        if lms_dict is None:
            frame_details.append({
                "frame_path": frame_web_path,
                "face_path": None,
                "face_detected": False,
                "seq_idx": seq_idx,
                "frame_num": fi
            })
            continue

        # Compute EAR + IOD
        right_pts = np.array([lms_dict[i] for i in RIGHT_EYE_106], dtype=float)
        left_pts  = np.array([lms_dict[i] for i in LEFT_EYE_106], dtype=float)
        raw_ear.append(
            (compute_EAR(right_pts) + compute_EAR(left_pts)) / 2.0)
        raw_iod.append(
            euclidean(lms_dict[RIGHT_EYE_OUTER_106],
                      lms_dict[LEFT_EYE_OUTER_106]))

        # Face crop from landmark bounds (identical to training)
        all_x = [v[0] for v in lms_dict.values()]
        all_y = [v[1] for v in lms_dict.values()]
        y1 = max(0, min(all_y) - 20)
        y2 = min(h, max(all_y) + 20)
        x1 = max(0, min(all_x) - 20)
        x2 = min(w, max(all_x) + 20)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save face crop
        face_filename = f"face_{seq_idx:03d}.jpg"
        face_save_path = os.path.join(faces_dir, face_filename)
        cv2.imwrite(face_save_path,
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        face_web_path = f"extracted_faces/{session_id}/{face_filename}"

        raw_crops.append(crop)
        per_frame.append((lms_dict, w, h, seq_idx))

        frame_details.append({
            "frame_path": frame_web_path,
            "face_path": face_web_path,
            "face_detected": True,
            "seq_idx": seq_idx,
            "frame_num": fi
        })

    if len(per_frame) < CFG.MIN_FRAMES:
        return {"error": "Not enough faces detected in the video."}

    # ── Video-level stats ─────────────────────────────────────────────
    blink_count, consec = 0, 0
    for ear in raw_ear:
        if ear < CFG.EAR_BLINK_THRESH:
            consec += 1
        else:
            if consec >= CFG.CONSEC_FRAMES:
                blink_count += 1
            consec = 0
    blink_rate = np.clip(
        blink_count / (duration_sec + 1e-6) / 0.5, 0, 2.0)
    iod_var = np.clip(
        float(np.std(raw_iod)) / (np.mean(raw_iod) + 1e-6)
        if raw_iod else 0.0, 0, 1.0)

    # ── GPU batch resize ──────────────────────────────────────────────
    resized_crops_np = []
    for start in range(0, len(raw_crops), CFG.RESIZE_BATCH):
        batch = raw_crops[start:start + CFG.RESIZE_BATCH]
        max_h = max(c.shape[0] for c in batch)
        max_w = max(c.shape[1] for c in batch)
        padded = np.zeros((len(batch), max_h, max_w, 3), dtype=np.uint8)
        for j, c in enumerate(batch):
            padded[j, :c.shape[0], :c.shape[1], :] = c
        resized = gpu_resize_batch(
            tf.constant(padded), CFG.IMG_SIZE, CFG.IMG_SIZE)
        resized_crops_np.append(resized.numpy())
    all_resized = np.concatenate(resized_crops_np, axis=0)

    # ── GPU batch image quality ───────────────────────────────────────
    quality_np = gpu_image_quality_batch(
        tf.constant(all_resized)).numpy()

    # ── Build 14-feature vectors (IDENTICAL to training) ──────────────
    faces_list = []
    meta_list  = []
    per_face_scores = []  # track per-face analysis

    for k, (lms_dict, img_w, img_h, seq_idx) in enumerate(per_frame):
        try:
            right_pts = np.array([lms_dict[i] for i in RIGHT_EYE_106], dtype=float)
            left_pts  = np.array([lms_dict[i] for i in LEFT_EYE_106], dtype=float)
            mouth_pts = np.array([lms_dict[i] for i in MOUTH_106], dtype=float)

            ear_l    = compute_EAR(left_pts)
            ear_r    = compute_EAR(right_pts)
            mar      = compute_MAR(mouth_pts)
            eye_dist = euclidean(lms_dict[RIGHT_EYE_OUTER_106],
                                 lms_dict[LEFT_EYE_OUTER_106])
            mouth_w  = euclidean(lms_dict[MOUTH_RIGHT_106],
                                 lms_dict[MOUTH_LEFT_106])
            face_h   = euclidean(lms_dict[RIGHT_EYE_OUTER_106],
                                 lms_dict[CHIN_106])
            nose_ch  = euclidean(lms_dict[NOSE_TIP_106],
                                 lms_dict[CHIN_106])
            symmetry = compute_symmetry(lms_dict)

            bright   = float(quality_np[k, 0])
            contrast = float(quality_np[k, 1])
            lap_var  = float(quality_np[k, 2])

            feature_vec = np.array([
                ear_l, ear_r, mar,
                eye_dist / (img_w + 1e-6),
                mouth_w  / (img_w + 1e-6),
                face_h   / (img_h + 1e-6),
                nose_ch  / (face_h + 1e-6),
                symmetry, bright, contrast, lap_var,
                blink_rate, iod_var,
                seq_idx / max(n_to_sample - 1, 1)
            ], dtype=np.float32)

            faces_list.append(all_resized[k])
            meta_list.append(np.clip(feature_vec, -5.0, 5.0))

            per_face_scores.append({
                "seq_idx": seq_idx,
                "symmetry": round(float(symmetry), 4),
                "ear_avg": round((ear_l + ear_r) / 2, 4),
                "brightness": round(bright, 4),
                "contrast": round(contrast, 4),
                "sharpness": round(lap_var, 4),
            })
        except Exception:
            continue

    valid_count = len(faces_list)
    if valid_count < CFG.MIN_FRAMES:
        return {"error": "Feature extraction failed — too few valid faces."}

    # ── Pad / truncate to N_FRAMES (identical to training) ────────────
    faces_arr = np.stack(faces_list, axis=0).astype(np.float32)
    meta_arr  = np.stack(meta_list, axis=0).astype(np.float32)

    n = valid_count
    if n < CFG.N_FRAMES:
        pad_f = np.full(
            (CFG.N_FRAMES - n, CFG.IMG_SIZE, CFG.IMG_SIZE, 3),
            -1.0, dtype=np.float32)
        pad_m = np.full(
            (CFG.N_FRAMES - n, CFG.FEATURE_COUNT),
            -1.0, dtype=np.float32)
        faces_arr = np.concatenate([faces_arr, pad_f], axis=0)
        meta_arr  = np.concatenate([meta_arr, pad_m], axis=0)

    valid_mask = np.zeros(CFG.N_FRAMES, dtype=np.float32)
    valid_mask[:min(n, CFG.N_FRAMES)] = 1.0

    faces_arr = faces_arr[:CFG.N_FRAMES]
    meta_arr  = meta_arr[:CFG.N_FRAMES]

    # ── Model Prediction ──────────────────────────────────────────────
    faces_batch = np.expand_dims(faces_arr, axis=0)   # (1, 32, 224, 224, 3)
    meta_batch  = np.expand_dims(meta_arr, axis=0)    # (1, 32, 14)
    mask_batch  = np.expand_dims(valid_mask, axis=0)   # (1, 32)

    prediction = model.predict(
        [faces_batch, meta_batch, mask_batch], verbose=0)
    fake_prob = float(prediction[0][0])
    real_prob = 1.0 - fake_prob
    is_fake   = fake_prob >= 0.5

    return {
        "error": None,
        "is_fake": is_fake,
        "verdict": "FAKE" if is_fake else "REAL",
        "fake_confidence": round(fake_prob * 100, 2),
        "real_confidence": round(real_prob * 100, 2),
        "total_frames_sampled": len(indexed_frames),
        "faces_detected": valid_count,
        "video_duration": round(duration_sec, 2),
        "video_fps": round(fps, 1),
        "video_resolution": f"{width}×{height}",
        "blink_rate": round(float(blink_rate), 4),
        "iod_variance": round(float(iod_var), 4),
        "frame_details": frame_details,
        "per_face_scores": per_face_scores,
        "session_id": session_id,
    }


# ═══════════════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return render_template("index.html", error="No video file uploaded.")

    file = request.files["video"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_file(file.filename):
        return render_template("index.html",
                               error="Unsupported format. Use MP4, AVI, MOV, MKV, or WebM.")

    # Save with unique session ID
    session_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{session_id}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    logger.info(f"Processing video: {filename} (session: {session_id})")
    t0 = time.time()

    result = process_video(save_path, session_id)

    elapsed = round(time.time() - t0, 2)
    result["processing_time"] = elapsed
    result["video_filename"] = file.filename

    logger.info(f"Done in {elapsed}s — verdict: {result.get('verdict', 'ERROR')}")

    if result.get("error"):
        return render_template("index.html", error=result["error"])

    return render_template("result.html", result=result)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API endpoint for programmatic access."""
    if "video" not in request.files:
        return jsonify({"error": "No video file in request."}), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported format."}), 400

    session_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{session_id}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    t0 = time.time()
    result = process_video(save_path, session_id)
    result["processing_time"] = round(time.time() - t0, 2)

    # Remove frame_details from API response (too large)
    result.pop("frame_details", None)
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
