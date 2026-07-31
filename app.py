import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO

# ── Page config ──
st.set_page_config(
    page_title="Rice Classifier",
    page_icon="🌾",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #facc15, #84cc16);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-top: 1rem;
    }
    .pred-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #facc15;
    }
    .pred-conf {
        font-size: 1.05rem;
        color: #cbd5e1;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #334155;
        border-radius: 14px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ──
st.markdown('<p class="main-title">🌾 Rice Grain Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and the CNN will identify the rice variety</p>', unsafe_allow_html=True)

# ── Class names, in the exact order from training ──
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

IMG_SIZE = 150  # matches this model's input shape (150, 150, 3)
CONFIDENCE_THRESHOLD = 60  # percent — below this, flag as "not confidently recognized"

# ── Load model ──
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_cnn_model.keras")

model = load_model()

# ── Preprocessing ──
# NOTE: This model has NO built-in Rescaling layer.
# It was trained on ImageDataGenerator(rescale=1./255), so we must divide by
# 255 ourselves here, or predictions will be meaningless.
def preprocess(img, img_size=IMG_SIZE):
    img = img.resize((img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── Main layout ──
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    input_method = st.radio("Choose input method", ["Upload a file", "Paste image URL"], horizontal=True)

    img = None

    if input_method == "Upload a file":
        uploaded_file = st.file_uploader("Upload a rice grain image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("Paste image URL here")
        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
            except requests.exceptions.RequestException as e:
                st.error(f"Couldn't fetch the image — check the URL. ({e})")
            except Exception:
                st.error("That URL didn't return a valid image.")

    if img is not None:
        st.image(img, caption="Input image", use_container_width=True)

with col2:
    if img is not None:
        with st.spinner("Analyzing..."):
            img_array = preprocess(img)
            predictions = model.predict(img_array)[0]

        top_idx = np.argsort(predictions)[::-1]
        best_idx = top_idx[0]
        predicted_class = CLASS_NAMES[best_idx]
        confidence = predictions[best_idx] * 100

        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class="result-card">
                <div class="pred-label" style="color:#f87171;">⚠️ Not confidently recognized</div>
                <div class="pred-conf">Closest guess was "{predicted_class}" at only {confidence:.2f}% confidence — too low to trust.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <div class="pred-label">🏆 {predicted_class}</div>
                <div class="pred-conf">Confidence: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### All predictions")
        df = pd.DataFrame({
            "Class": [CLASS_NAMES[i] for i in top_idx],
            "Confidence (%)": [round(float(predictions[i]) * 100, 2) for i in top_idx]
        }).set_index("Class")
        st.bar_chart(df, horizontal=True)
    else:
        st.markdown("""
        <div class="result-card">
            <div class="pred-label" style="color:#64748b;">No prediction yet</div>
            <div class="pred-conf">Upload an image or paste a URL to see results here.</div>
        </div>
        """, unsafe_allow_html=True)
