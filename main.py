import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline


APP_DIR = Path(__file__).parent
LANDMARKER_PATH = APP_DIR / "pose_landmarker_full.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker/float16/latest/pose_landmarker_full.task"
)
MODEL_ID = "dima806/yoga_pose_image_classification"
POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


st.set_page_config(page_title="Yoga Pose Classifier", page_icon="Y", layout="wide")


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("image-classification", model=MODEL_ID, top_k=5)


@st.cache_resource(show_spinner=False)
def load_landmarker():
    if not LANDMARKER_PATH.exists():
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)

    base_options = mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def normalize_label(label):
    cleaned = label.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "downdog": "downward-dog",
        "downwarddog": "downward-dog",
        "downward-facing-dog": "downward-dog",
        "mountain": "standing-mountain",
        "tadasana": "standing-mountain",
    }
    return aliases.get(cleaned, cleaned)


def point(landmarks, index, width, height):
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def angle_degrees(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def rule_score(landmarks, width, height, rules):
    scores = []
    details = []

    for rule in rules:
        name, triplets, target, tolerance = rule[:4]
        reducer = rule[4] if len(rule) > 4 else "mean"
        values = []
        for a, b, c in triplets:
            measured = angle_degrees(
                point(landmarks, a, width, height),
                point(landmarks, b, width, height),
                point(landmarks, c, width, height),
            )
            if measured is not None:
                values.append(measured)

        if not values:
            continue

        if reducer == "min":
            measured = float(np.min(values))
        elif reducer == "max":
            measured = float(np.max(values))
        else:
            measured = float(np.mean(values))
        deviation = abs(measured - target)
        score = max(0.0, 1.0 - deviation / tolerance)
        scores.append(score)
        details.append(
            {
                "Check": name,
                "Measured": f"{measured:.0f} deg",
                "Target": f"{target:.0f} deg",
                "Score": f"{int(score * 100)}",
            }
        )

    if not scores:
        return None, []

    return int(np.mean(scores) * 100), details


POSE_RULES = {
    "standing-mountain": [
        ("legs long", [(23, 25, 27), (24, 26, 28)], 176, 18),
        ("hips stacked", [(11, 23, 25), (12, 24, 26)], 170, 28),
        ("torso upright", [(23, 11, 13), (24, 12, 14)], 170, 35),
    ],
    "tree": [
        ("standing leg long", [(23, 25, 27), (24, 26, 28)], 172, 22, "max"),
        ("raised knee folded", [(23, 25, 27), (24, 26, 28)], 60, 40, "min"),
        ("torso tall", [(23, 11, 13), (24, 12, 14)], 165, 40),
    ],
    "warrior": [
        ("front knee bent", [(23, 25, 27), (24, 26, 28)], 115, 45, "min"),
        ("back leg strong", [(23, 25, 27), (24, 26, 28)], 165, 35, "max"),
        ("arms extended", [(13, 11, 23), (14, 12, 24)], 92, 45),
    ],
    "triangle": [
        ("legs long", [(23, 25, 27), (24, 26, 28)], 170, 25),
        ("side body angle", [(11, 23, 25), (12, 24, 26)], 105, 45),
        ("top arm line", [(13, 11, 23), (14, 12, 24)], 95, 45),
    ],
    "downward-dog": [
        ("inverted hip angle", [(11, 23, 25), (12, 24, 26)], 80, 35),
        ("legs lengthened", [(23, 25, 27), (24, 26, 28)], 160, 35),
        ("arms lengthened", [(11, 13, 15), (12, 14, 16)], 160, 35),
    ],
    "cobra": [
        ("elbows soft", [(11, 13, 15), (12, 14, 16)], 145, 45),
        ("front body lifted", [(11, 23, 25), (12, 24, 26)], 130, 45),
    ],
    "child": [
        ("folded hips", [(11, 23, 25), (12, 24, 26)], 55, 35),
        ("knees folded", [(23, 25, 27), (24, 26, 28)], 45, 35),
    ],
    "bridge": [
        ("knees bent", [(23, 25, 27), (24, 26, 28)], 95, 35),
        ("hips open", [(11, 23, 25), (12, 24, 26)], 155, 35),
    ],
    "pigeon": [
        ("front leg folded", [(23, 25, 27), (24, 26, 28)], 65, 45, "min"),
        ("torso angle", [(11, 23, 25), (12, 24, 26)], 125, 50),
    ],
}


def get_pose_landmarks(image_array):
    landmarker = load_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def draw_pose(image_array, landmarks):
    output = image_array.copy()
    height, width = output.shape[:2]

    for start, end in POSE_CONNECTIONS:
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        a = landmarks[start]
        b = landmarks[end]
        if a.visibility < 0.25 or b.visibility < 0.25:
            continue
        cv2.line(
            output,
            (int(a.x * width), int(a.y * height)),
            (int(b.x * width), int(b.y * height)),
            (63, 142, 252),
            3,
        )

    for landmark in landmarks:
        if landmark.visibility < 0.25:
            continue
        cv2.circle(
            output,
            (int(landmark.x * width), int(landmark.y * height)),
            4,
            (20, 184, 166),
            -1,
        )

    return output


def score_prediction(predictions):
    top_score = float(predictions[0]["score"])
    next_score = float(predictions[1]["score"]) if len(predictions) > 1 else 0.0
    margin = max(0.0, top_score - next_score)
    certainty = int(round((0.72 * top_score + 0.28 * margin) * 100))
    return certainty, int(round(top_score * 100)), int(round(margin * 100))


def feedback_for_score(score):
    if score >= 85:
        return "Strong match"
    if score >= 65:
        return "Likely match"
    if score >= 45:
        return "Uncertain"
    return "Low confidence"


st.title("Yoga Pose Classifier")
st.caption(
    "Upload or capture a yoga pose image. The app predicts the pose, scores model certainty, "
    "and estimates form from body landmarks when a person is visible."
)

with st.sidebar:
    st.subheader("Model")
    st.write(f"Hugging Face: `{MODEL_ID}`")
    st.write("Pose form scoring: MediaPipe Pose Landmarker")
    show_checks = st.toggle("Show alignment checks", value=True)

left, right = st.columns([1, 1], gap="large")

with left:
    input_mode = st.radio("Image source", ["Upload image", "Take photo"], horizontal=True)
    image_file = None
    if input_mode == "Upload image":
        image_file = st.file_uploader("Choose an image", type=("jpg", "jpeg", "png", "webp"))
    else:
        image_file = st.camera_input("Take a photo")

if image_file is None:
    with right:
        st.info("Add a clear full-body yoga pose image to start.")
    st.stop()

pil_image = Image.open(image_file).convert("RGB")
image_array = np.array(pil_image)
height, width = image_array.shape[:2]

with st.spinner("Loading model and reading the pose..."):
    classifier = load_classifier()
    predictions = classifier(pil_image)
    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    landmarks = get_pose_landmarks(image_array)

best_label = predictions[0]["label"]
pose_key = normalize_label(best_label)
certainty_score, model_confidence, ambiguity_margin = score_prediction(predictions)

form_score = None
alignment_details = []
annotated = image_array

if landmarks:
    annotated = draw_pose(image_array, landmarks)
    if pose_key in POSE_RULES:
        form_score, alignment_details = rule_score(landmarks, width, height, POSE_RULES[pose_key])

if form_score is None:
    final_score = certainty_score
else:
    final_score = int(round(0.65 * certainty_score + 0.35 * form_score))

with left:
    st.image(annotated, caption="Analyzed image", use_container_width=True)

with right:
    st.subheader(best_label)
    st.progress(final_score / 100)
    st.metric("Overall accuracy score", f"{final_score}/100", feedback_for_score(final_score))

    metric_cols = st.columns(3)
    metric_cols[0].metric("Model confidence", f"{model_confidence}%")
    metric_cols[1].metric("Top-2 margin", f"{ambiguity_margin}%")
    if form_score is None:
        metric_cols[2].metric("Form score", "N/A")
    else:
        metric_cols[2].metric("Form score", f"{form_score}%")

    if not landmarks:
        st.warning("No full-body pose landmarks were found, so scoring uses model certainty only.")
    elif pose_key not in POSE_RULES:
        st.info("Pose landmarks were detected, but no alignment rule exists for this label yet.")

    st.subheader("Top predictions")
    for prediction in predictions[:5]:
        label = prediction["label"]
        score = float(prediction["score"])
        st.write(f"{label}: {score:.1%}")
        st.progress(score)

    if show_checks and alignment_details:
        st.subheader("Alignment checks")
        st.dataframe(alignment_details, hide_index=True, use_container_width=True)

st.caption(
    "Scoring combines the classifier confidence and its gap over the runner-up prediction. "
    "When landmarks are available, pose-specific angle checks contribute to the overall score."
)
