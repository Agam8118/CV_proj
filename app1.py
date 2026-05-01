import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image

# Import your functions from your script
from dual_branch_classifier import (
    standardize,
    extract_features,
    LABEL_NAMES
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Dual-Branch Defect Classifier",
    layout="wide"
)

st.title("🔍 Dual-Branch Surface Defect Classifier")
st.markdown("Detect **Scratch vs Patch vs Clean Metal** using classical CV")

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "output/dual_branch_rf.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Run training first!")
        return None
    return joblib.load(MODEL_PATH)

model_bundle = load_model()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Controls")
show_features = st.sidebar.checkbox("Show Feature Details", True)

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload Surface Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    # ===============================
    # PREPROCESS
    # ===============================
    temp_path = "temp.jpg"
    image.save(temp_path)

    img_std = standardize(temp_path)

    with col2:
        st.image(img_std, caption="Standardized (128x128 + HE)", use_container_width=True)

    # ===============================
    # FEATURE EXTRACTION
    # ===============================
    struct_feats, lbp = extract_features(img_std)

    # ===============================
    # PCA + MODEL
    # ===============================
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]
    pca = model_bundle["pca"]

    lbp_pca = pca.transform(lbp.reshape(1, -1))
    features = np.hstack([struct_feats.reshape(1, -1), lbp_pca])
    features_scaled = scaler.transform(features)

    probs = model.predict_proba(features_scaled)[0]

    full_probs = np.zeros(len(LABEL_NAMES))
    for i, cls in enumerate(model.classes_):
        full_probs[cls] = probs[i]

    pred_class = int(np.argmax(full_probs))

    # ===============================
    # RESULT DISPLAY
    # ===============================
    st.subheader("📊 Prediction Result")

    st.success(f"Prediction: **{LABEL_NAMES[pred_class]}**")
    st.metric("Confidence", f"{full_probs[pred_class]*100:.2f}%")

    # ===============================
    # PROBABILITY BAR
    # ===============================
    st.subheader("Class Probabilities")

    prob_dict = {
        LABEL_NAMES[i]: float(full_probs[i])
        for i in range(len(LABEL_NAMES))
    }

    st.bar_chart(prob_dict)

    # ===============================
    # FEATURE DETAILS
    # ===============================
    if show_features:
        st.subheader("🔬 Feature Insights")

        st.write(f"**Hough Line Count:** {struct_feats[0]:.0f}")
        st.write(f"**Edge Density:** {struct_feats[1]:.4f}")
        st.write(f"**Max Aspect Ratio:** {struct_feats[2]:.2f}")
        st.write(f"**GLCM Homogeneity:** {struct_feats[3]:.4f}")
        st.write(f"**GLCM Contrast:** {struct_feats[4]:.4f}")
        st.write(f"**Pixel Variance:** {struct_feats[5]:.2f}")

        st.write("**Top PCA Components (first 5):**")
        st.write(lbp_pca[0][:5])

    # ===============================
    # EDGE VISUALIZATION
    # ===============================
    st.subheader("🧠 Edge Detection View")

    edges = cv2.Canny(img_std, 60, 140)
    st.image(edges, caption="Canny Edge Map", use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("Built with Classical Computer Vision | No Deep Learning 🚀")