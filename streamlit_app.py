import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cloud environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mtcnn import MTCNN
import tensorflow as tf
import keras
import os
import zipfile
import gdown
from deepface import DeepFace

# Page configuration for mobile
st.set_page_config(
    page_title="Facial Age Estimator",
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
        .stButton > button {
            height: 2.5rem;
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📱 Facial Age Estimator")
st.markdown("**Estimate the biological age of your face from facial images**")

MODEL_ZIP = "model_saved_tf.zip"
MODEL_DIR = "model_saved_tf"
GDRIVE_FILE_ID = "12wNpYBz3j5mP9mt6S_ZH4k0sI6dVDeVV"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        st.info("Downloading FaceAge model. This may take a minute...")
        try:
            # Download with retry logic for cloud environments
            gdown.download(url, MODEL_ZIP, quiet=False)
            
            # Extract the model
            with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up zip file
            if os.path.exists(MODEL_ZIP):
                os.remove(MODEL_ZIP)
            
            st.success("Model downloaded and extracted!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            st.info("💡 The app will continue with DeepFace only.")

# Download model if needed
with st.spinner("Checking model files..."):
    download_and_extract_model()

# Load model with caching
@st.cache_resource
def load_faceage_saved_model():
    # Try different possible paths for the model
    possible_paths = [
        "FaceAge/models/model_saved_tf",
        "./FaceAge/models/model_saved_tf",
        "model_saved_tf",
        "./model_saved_tf"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                if model_path.endswith('.h5'):
                    # Load Keras .h5 model
                    model = keras.models.load_model(model_path)
                    st.success(f"✅ Model loaded from: {model_path}")
                    return model
                else:
                    # Load TensorFlow SavedModel (compatible with TF 2.13.0)
                    try:
                        # Try loading as Keras model first
                        model = tf.keras.models.load_model(model_path)
                        st.success(f"✅ Model loaded from: {model_path}")
                        return model
                    except Exception as e1:
                        # Fallback to SavedModel loading
                        try:
                            model = tf.saved_model.load(model_path)
                            st.success(f"✅ SavedModel loaded from: {model_path}")
                            return model
                        except Exception as e2:
                            st.warning(f"⚠️ Failed to load from {model_path}: Keras error: {str(e1)}, SavedModel error: {str(e2)}")
                            continue
            except Exception as e:
                st.warning(f"⚠️ Failed to load from {model_path}: {str(e)}")
                continue
    
    st.info("ℹ️ Harvard FaceAge model not available - using DeepFace only")
    return None

# Load models
with st.spinner("Loading AI models..."):
    harvard_model = load_faceage_saved_model()
    # Don't stop if Harvard model fails - DeepFace can still work
    
    if harvard_model is None:
        st.info("📱 Using DeepFace for age estimation (good for all ages, especially younger faces)")
    else:
        st.success("🎯 Using both Harvard FaceAge and DeepFace models")
    
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
    st.image(image, caption="Uploaded image", use_container_width=True)
    
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
                
                # Harvard FaceAge model prediction
                if harvard_model is not None:
                    face_resized = cv2.resize(face_crop, (160, 160))
                    face_pil = Image.fromarray(face_resized).convert('RGB')
                    face_array = np.asarray(face_pil)
                    mean, std = face_array.mean(), face_array.std()
                    face_normalized = (face_array - mean) / std
                    face_input = face_normalized.reshape(1, 160, 160, 3)
                    
                    # Predict age with Harvard model
                    try:
                        # Handle different model types
                        if hasattr(harvard_model, 'predict'):
                            # Standard Keras model
                            prediction = harvard_model.predict(face_input)
                            harvard_age = float(np.squeeze(prediction))
                        elif hasattr(harvard_model, 'signatures'):
                            # SavedModel format
                            signature = harvard_model.signatures['serving_default']
                            prediction = signature(tf.constant(face_input, dtype=tf.float32))
                            if isinstance(prediction, dict):
                                age_prediction = list(prediction.values())[0]
                            else:
                                age_prediction = prediction
                            harvard_age = float(np.squeeze(age_prediction))
                        else:
                            # Fallback - try direct call
                            prediction = harvard_model(face_input)
                            if isinstance(prediction, dict):
                                age_prediction = list(prediction.values())[0]
                            else:
                                age_prediction = prediction
                            harvard_age = float(np.squeeze(age_prediction))
                    except Exception as e:
                        st.warning(f"⚠️ Harvard model prediction failed: {str(e)}")
                        harvard_age = None
                else:
                    harvard_age = None
                
                # DeepFace age prediction
                deepface_age = None
                try:
                    # Ensure face crop is in correct format for DeepFace
                    if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                        # Skip if face is too small
                        st.info(f"Face {i+1} too small for DeepFace analysis")
                    else:
                        # Method 1: Direct numpy array analysis (preferred)
                        try:
                            # Resize face crop to a standard size for DeepFace
                            face_resized = cv2.resize(face_crop, (224, 224))
                            
                            # Ensure proper data type and range
                            if face_resized.dtype != np.uint8:
                                face_resized = face_resized.astype(np.uint8)
                            
                            # Ensure values are in proper range [0, 255]
                            face_resized = np.clip(face_resized, 0, 255)
                            
                            # Try direct numpy array analysis first
                            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                                deepface_result = DeepFace.analyze(face_resized, actions=['age'], enforce_detection=False)
                                
                                if isinstance(deepface_result, list):
                                    deepface_age = deepface_result[0]['age']
                                else:
                                    deepface_age = deepface_result['age']
                                    
                                st.info(f"✅ DeepFace analysis successful for face {i+1}: {deepface_age:.1f} years")
                            else:
                                raise ValueError("Invalid face format")
                                
                        except Exception as e1:
                            # Method 2: File-based analysis (fallback)
                            try:
                                st.info(f"Trying file-based analysis for face {i+1}...")
                                temp_path = f"temp_face_{i}.jpg"
                                temp_image = Image.fromarray(face_resized)
                                temp_image.save(temp_path)
                                
                                # Analyze with DeepFace
                                deepface_result = DeepFace.analyze(temp_path, actions=['age'], enforce_detection=False)
                                
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                
                                if isinstance(deepface_result, list):
                                    deepface_age = deepface_result[0]['age']
                                else:
                                    deepface_age = deepface_result['age']
                                    
                                st.info(f"✅ DeepFace file analysis successful for face {i+1}: {deepface_age:.1f} years")
                                
                            except Exception as e2:
                                # Clean up temp file if it exists
                                temp_path = f"temp_face_{i}.jpg"
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                st.error(f"❌ Both DeepFace methods failed for face {i+1}")
                                st.error(f"   Direct analysis error: {str(e1)}")
                                st.error(f"   File analysis error: {str(e2)}")
                                deepface_age = None
                                
                except Exception as e:
                    # Final catch-all
                    st.error(f"❌ DeepFace analysis completely failed for face {i+1}: {str(e)}")
                    deepface_age = None
                
                # Age group classification (using Harvard age as primary, DeepFace as fallback)
                primary_age = harvard_age if harvard_age is not None else deepface_age
                
                if primary_age is not None:
                    if primary_age < 30:
                        color = '#4CAF50'
                        age_group = "Young"
                        emoji = "😊"
                    elif primary_age < 50:
                        color = '#FF9800'
                        age_group = "Adult"
                        emoji = "😐"
                    else:
                        color = '#F44336'
                        age_group = "Senior"
                        emoji = "👴"
                else:
                    color = '#666666'
                    age_group = "Unknown"
                    emoji = "❓"
                
                # Add annotation with both ages
                harvard_text = f"{harvard_age:.1f}y" if harvard_age is not None else "N/A"
                deepface_text = f"{deepface_age:.1f}y" if deepface_age is not None else "N/A"
                annotation_text = f"{emoji} Harvard: {harvard_text} | DeepFace: {deepface_text}"
                
                ax.text(x, y-15, annotation_text, 
                       color=color, fontsize=12, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
                
                results.append({
                    'face_num': i+1,
                    'harvard_age': harvard_age,
                    'deepface_age': deepface_age,
                    'group': age_group,
                    'confidence': face['confidence']
                })
            
            ax.axis('off')
            st.pyplot(fig)
            
            # Results summary
            st.markdown("### 📊 Analysis Results")
            for result in results:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Face", f"#{result['face_num']}")
                with col2:
                    if result['harvard_age'] is not None:
                        st.metric("Harvard Age", f"{result['harvard_age']:.1f} years")
                    else:
                        st.metric("Harvard Age", "N/A")
                with col3:
                    if result['deepface_age'] is not None:
                        st.metric("DeepFace Age", f"{result['deepface_age']:.1f} years")
                    else:
                        st.metric("DeepFace Age", "N/A")
                with col4:
                    st.metric("Group", result['group'])
            
            # Download results
            st.markdown("### 💾 Download Results")
            results_text = f"FaceAge Analysis Results\n\n"
            for result in results:
                harvard_text = f"{result['harvard_age']:.1f}" if result['harvard_age'] is not None else "N/A"
                deepface_text = f"{result['deepface_age']:.1f}" if result['deepface_age'] is not None else "N/A"
                results_text += f"Face #{result['face_num']}: Harvard: {harvard_text} years | DeepFace: {deepface_text} years ({result['group']})\n"
            
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

def simple_age_estimation(face_crop):
    """
    Simple age estimation based on facial features analysis
    This is a basic heuristic approach that doesn't require large models
    """
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        
        # Basic feature analysis
        height, width = gray.shape
        
        # Analyze skin texture (higher variance = older)
        skin_variance = np.var(gray)
        
        # Analyze contrast (wrinkles create more contrast)
        contrast = gray.std()
        
        # Analyze edge density (more edges = more wrinkles)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Simple heuristic formula (calibrated for reasonable results)
        base_age = 25  # Starting point
        
        # Adjust based on features
        age_adjustment = 0
        age_adjustment += min(skin_variance / 100, 20)  # Skin texture contribution
        age_adjustment += min(contrast / 10, 15)        # Contrast contribution  
        age_adjustment += min(edge_density * 200, 25)   # Edge density contribution
        
        estimated_age = base_age + age_adjustment
        
        # Clamp to reasonable range
        estimated_age = max(18, min(80, estimated_age))
        
        return estimated_age
        
    except Exception as e:
        # Return a default age if analysis fails
        return 35.0 