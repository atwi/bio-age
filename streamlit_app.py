import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mtcnn import MTCNN
import tensorflow as tf
import keras
from keras.layers import TFSMLayer
import os
import zipfile
import gdown

# Page configuration for mobile
st.set_page_config(
    page_title="FaceAge Estimator",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        border-radius: 10px;
        background-color: #FF6B6B;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #FF5252;
    }
    .uploadedFile {
        border-radius: 10px;
        padding: 1rem;
        background-color: #f0f2f6;
    }
    .stMarkdown {
        font-size: 1.1rem;
    }
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📱 FaceAge Estimator")
st.markdown("**Biological age estimation from facial images**")

MODEL_ZIP = "model_saved_tf.zip"
MODEL_DIR = "model_saved_tf"
GDRIVE_FILE_ID = "12wNpYBz3j5mP9mt6S_ZH4k0sI6dVDeVV"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        st.info("Downloading FaceAge model. This may take a minute...")
        gdown.download(url, MODEL_ZIP, quiet=False)
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(MODEL_ZIP)
        st.success("Model downloaded and extracted!")

# Download model if needed
with st.spinner("Checking model files..."):
    download_and_extract_model()

# Load model with caching
@st.cache_resource
def load_faceage_saved_model():
    # Try different possible paths for the model
    possible_paths = [
        "model_saved_tf",
        "FaceAge_Project/model_saved_tf",
        "./model_saved_tf",
        "./FaceAge_Project/model_saved_tf"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                model = TFSMLayer(model_path, call_endpoint="serving_default")
                st.success(f"✅ Model loaded from: {model_path}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Failed to load from {model_path}: {str(e)}")
                continue
    
    st.error("❌ Could not load model from any location")
    return None

# Load models
with st.spinner("Loading AI models..."):
    model = load_faceage_saved_model()
    if model is None:
        st.error("Failed to load FaceAge model. Please check the model files.")
        st.stop()
    
    detector = MTCNN()

# Main interface
st.markdown("### 📸 Upload a photo or take a new one")

# Add tabs for upload and camera
upload_tab, camera_tab = st.tabs(["Upload Image", "Take Photo"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of a face"
    )

with camera_tab:
    camera_image = st.camera_input("Take a photo")

# Use whichever is provided
image_source = uploaded_file if uploaded_file is not None else camera_image

if image_source is not None:
    # Display original image
    st.markdown("### 📷 Original Image")
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    
    # Process image
    with st.spinner("Analyzing face..."):
        img_np = np.array(image)
        faces = detector.detect_faces(img_np)
        
        if not faces:
            st.error("❌ No face detected in this image.")
            st.info("💡 Try uploading a clearer photo with a visible face.")
        else:
            st.success(f"✅ Detected {len(faces)} face(s)")
            
            # Create results display
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_np)
            
            results = []
            for i, face in enumerate(faces):
                x, y, w, h = face['box']
                x, y = abs(x), abs(y)
                x2, y2 = x + w, y + h
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor='#FF6B6B', facecolor='none')
                ax.add_patch(rect)
                
                # Process face for age estimation
                face_crop = img_np[y:y2, x:x2]
                face_resized = cv2.resize(face_crop, (160, 160))
                face_pil = Image.fromarray(face_resized).convert('RGB')
                face_array = np.asarray(face_pil)
                mean, std = face_array.mean(), face_array.std()
                face_normalized = (face_array - mean) / std
                face_input = face_normalized.reshape(1, 160, 160, 3)
                
                # Predict age
                prediction = model(face_input)
                if isinstance(prediction, dict):
                    age_prediction = list(prediction.values())[0]
                else:
                    age_prediction = prediction
                estimated_age = float(np.squeeze(age_prediction))
                
                # Age group classification
                if estimated_age < 30:
                    color = '#4CAF50'
                    age_group = "Young"
                    emoji = "😊"
                elif estimated_age < 50:
                    color = '#FF9800'
                    age_group = "Adult"
                    emoji = "😐"
                else:
                    color = '#F44336'
                    age_group = "Senior"
                    emoji = "👴"
                
                # Add annotation
                ax.text(x, y-15, f"{emoji} {estimated_age:.1f}y ({age_group})", 
                       color=color, fontsize=14, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
                
                results.append({
                    'face_num': i+1,
                    'age': estimated_age,
                    'group': age_group,
                    'confidence': face['confidence']
                })
            
            ax.axis('off')
            st.pyplot(fig)
            
            # Results summary
            st.markdown("### 📊 Analysis Results")
            for result in results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Face", f"#{result['face_num']}")
                with col2:
                    st.metric("Age", f"{result['age']:.1f} years")
                with col3:
                    st.metric("Group", result['group'])
            
            # Download results
            st.markdown("### 💾 Download Results")
            results_text = f"FaceAge Analysis Results\n\n"
            for result in results:
                results_text += f"Face #{result['face_num']}: {result['age']:.1f} years ({result['group']})\n"
            
            st.download_button(
                label="📥 Download Results",
                data=results_text,
                file_name="faceage_results.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🔬 Powered by AI • 📱 Mobile Optimized • 🔒 Privacy Focused</p>
    <p>This app estimates biological age from facial features using machine learning.</p>
</div>
""", unsafe_allow_html=True) 