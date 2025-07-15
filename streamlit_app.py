import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN

# Cloud-specific optimizations
import os
import sys
import gc  # Garbage collection for memory management
import psutil  # Memory monitoring
import warnings
warnings.filterwarnings('ignore')

# Detect cloud environment early
IS_CLOUD = (
    'STREAMLIT_CLOUD' in os.environ or 
    'HOSTNAME' in os.environ or 
    'DYNO' in os.environ or
    'RAILWAY_ENVIRONMENT' in os.environ
)

# Suppress TensorFlow warnings aggressively for cloud
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if IS_CLOUD:
    # Cloud-specific environment settings
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    import keras
    import zipfile
    import gdown
    import time
    from deepface import DeepFace
    
    # Verify versions for cloud deployment
    tf_version = tf.__version__
    keras_version = keras.__version__
    
    # Only show version warnings locally, not in cloud
    if not IS_CLOUD:
        if not tf_version.startswith('2.13'):
            st.warning(f"⚠️ TensorFlow version {tf_version} detected. Expected 2.13.x")
        
        if not keras_version.startswith('2.13'):
            st.warning(f"⚠️ Keras version {keras_version} detected. Expected 2.13.x")
        
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Please check your requirements.txt file.")
    st.stop()

# Page configuration for mobile
st.set_page_config(
    page_title="Facial Age Estimator",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cloud environment info
if IS_CLOUD:
    st.info("🌟 Running on Streamlit Cloud with Harvard + DeepFace models")
    
    # Memory monitoring for cloud
    try:
        memory_info = psutil.virtual_memory()
        st.caption(f"Memory: {memory_info.percent}% used")
    except:
        pass
else:
    st.info("🖥️ Running locally with Harvard + DeepFace models")

# Streamlined CSS for cloud
st.markdown("""
<style>
    .main { padding: 0.5rem; }
    .stButton > button { width: 100%; height: 3rem; border-radius: 10px; }
    @media (max-width: 768px) { .main { padding: 0.25rem; } }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📱 Facial Age Estimator")
st.markdown("**Estimate biological age from facial features**")

# Cloud-optimized model handling
MODEL_ZIP = "model_saved_tf.zip"
MODEL_DIR = "model_saved_tf"
GDRIVE_FILE_ID = "12wNpYBz3j5mP9mt6S_ZH4k0sI6dVDeVV"

@st.cache_resource
def download_harvard_model():
    """Download Harvard model with cloud optimizations and enhanced error handling"""
    if not os.path.exists(MODEL_DIR):
        try:
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            
            # Enhanced progress indicator for cloud
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing download...")
            progress_bar.progress(10)
            
            try:
                status_text.text("📥 Downloading Harvard model (85MB)...")
                progress_bar.progress(30)
                
                # Use gdown with built-in timeout handling
                gdown.download(url, MODEL_ZIP, quiet=True)
                
                status_text.text("📦 Extracting model...")
                progress_bar.progress(70)
                
                with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
                    zip_ref.extractall(".")
                
                status_text.text("🧹 Cleaning up...")
                progress_bar.progress(90)
                
                if os.path.exists(MODEL_ZIP):
                    os.remove(MODEL_ZIP)
                
                progress_bar.progress(100)
                status_text.text("✅ Harvard model ready!")
                
                # Clean up progress indicators
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                return True
                
            except Exception as download_error:
                progress_bar.empty()
                status_text.empty()
                st.warning(f"⚠️ Harvard model download failed: {str(download_error)}. Using DeepFace only.")
                return False
                     
        except Exception as e:
            st.warning(f"⚠️ Harvard model download failed: {str(e)}. Using DeepFace only.")
            return False
    
    return True

@st.cache_resource
def load_harvard_model():
    """Load Harvard model with enhanced cloud compatibility"""
    # Try to download first
    if not download_harvard_model():
        st.info("ℹ️ Harvard model unavailable - using DeepFace only")
        return None
    
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
                # Cloud-optimized loading with timeout
                with st.spinner(f"🔄 Loading Harvard model from {model_path}..."):
                    if model_path.endswith('.h5'):
                        # Load Keras .h5 model
                        model = keras.models.load_model(model_path)
                        st.success(f"✅ Harvard model loaded from: {model_path}")
                        return model
                    else:
                        # Load TensorFlow SavedModel (compatible with TF 2.13.0)
                        try:
                            # Try loading as Keras model first
                            model = tf.keras.models.load_model(model_path)
                            st.success(f"✅ Harvard model loaded from: {model_path}")
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
    
    st.warning("⚠️ Harvard FaceAge model not available - using DeepFace only")
    return None

@st.cache_resource
def init_face_detector():
    """Initialize face detector with error handling"""
    try:
        detector = MTCNN()
        return detector
    except Exception as e:
        st.error(f"❌ Face detector failed: {str(e)}")
        return None

@st.cache_resource
def test_deepface_cloud():
    """Test DeepFace with cloud-specific settings"""
    try:
        # Create minimal test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Cloud-optimized DeepFace call
        result = DeepFace.analyze(
            test_img,
            actions=['age'],
            enforce_detection=False,
            silent=True
        )
        
        # Force garbage collection
        del test_img, result
        gc.collect()
        
        return True
        
    except Exception as e:
        st.error(f"❌ DeepFace test failed: {str(e)}")
        return False

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.detector = None
    st.session_state.harvard_model = None
    st.session_state.deepface_ok = False

def lazy_load_models():
    """Lazy load models only when user uploads an image"""
    if st.session_state.models_loaded:
        return True
    
    with st.spinner("🔄 Loading AI models... This may take a moment."):
        try:
            # Step 1: Face detector (essential)
            st.session_state.detector = init_face_detector()
            if st.session_state.detector is None:
                st.error("❌ Face detection failed - app cannot continue")
                return False
            
            # Step 2: Test DeepFace (essential)
            st.session_state.deepface_ok = test_deepface_cloud()
            if not st.session_state.deepface_ok:
                st.error("❌ DeepFace failed - app cannot continue")
                return False
            
            # Step 3: Harvard model (optional but now enabled for cloud)
            st.session_state.harvard_model = load_harvard_model()
            
            st.session_state.models_loaded = True
            
            # Success message
            models_msg = "DeepFace"
            if st.session_state.harvard_model:
                models_msg += " + Harvard"
            st.success(f"✅ Ready! Using {models_msg} models")
            
            # Memory cleanup
            gc.collect()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            return False

# Main interface
st.markdown("### 📸 Upload Your Photo")

# Simplified upload (cloud-friendly)
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo with visible face(s)"
)

if uploaded_file is not None:
    # Lazy load models only when user uploads an image
    if not lazy_load_models():
        st.error("❌ Failed to load models. Please refresh the page.")
        st.stop()
    
    # Display image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your photo", use_container_width=True)
        
        # Process with cloud optimizations
        with st.spinner("🔍 Analyzing faces..."):
            img_array = np.array(image)
            
            # Face detection
            faces = st.session_state.detector.detect_faces(img_array)
            
            if not faces:
                st.error("❌ No face detected. Please try a clearer photo.")
            else:
                st.success(f"✅ Found {len(faces)} face(s)")
                
                # Process each face
                for i, face in enumerate(faces):
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    
                    # Extract face
                    face_crop = img_array[y:y+h, x:x+w]
                    
                    # Try Harvard model first (now enabled for cloud)
                    harvard_age = None
                    if st.session_state.harvard_model is not None:
                        try:
                            # Prepare face for Harvard model
                            face_resized_harvard = cv2.resize(face_crop, (160, 160))
                            face_pil = Image.fromarray(face_resized_harvard).convert('RGB')
                            face_array = np.asarray(face_pil)
                            mean, std = face_array.mean(), face_array.std()
                            # Prevent division by zero
                            if std == 0:
                                std = 1
                            face_normalized = (face_array - mean) / std
                            face_input = face_normalized.reshape(1, 160, 160, 3)
                            
                            # Predict age with Harvard model
                            try:
                                # Handle different model types
                                if hasattr(st.session_state.harvard_model, 'predict'):
                                    # Standard Keras model
                                    prediction = st.session_state.harvard_model.predict(face_input, verbose=0)
                                    harvard_age = float(np.squeeze(prediction))
                                elif hasattr(st.session_state.harvard_model, 'signatures'):
                                    # SavedModel format
                                    signature = st.session_state.harvard_model.signatures['serving_default']
                                    prediction = signature(tf.constant(face_input, dtype=tf.float32))
                                    if isinstance(prediction, dict):
                                        age_prediction = list(prediction.values())[0]
                                    else:
                                        age_prediction = prediction
                                    harvard_age = float(np.squeeze(age_prediction))
                                else:
                                    # Fallback - try direct call
                                    prediction = st.session_state.harvard_model(face_input)
                                    if isinstance(prediction, dict):
                                        age_prediction = list(prediction.values())[0]
                                    else:
                                        age_prediction = prediction
                                    harvard_age = float(np.squeeze(age_prediction))
                                
                                # Clamp age to reasonable range
                                harvard_age = max(0, min(120, harvard_age))
                                
                            except Exception as e:
                                st.warning(f"⚠️ Harvard model prediction failed for face {i+1}: {str(e)}")
                                harvard_age = None
                        except Exception as e:
                            st.warning(f"⚠️ Harvard model preprocessing failed for face {i+1}: {str(e)}")
                            harvard_age = None
                    
                    # DeepFace analysis (cloud-optimized)
                    deepface_age = None
                    try:
                        if face_crop.shape[0] > 20 and face_crop.shape[1] > 20:
                            # Resize for consistency
                            face_resized = cv2.resize(face_crop, (224, 224))
                            face_resized = np.clip(face_resized, 0, 255).astype(np.uint8)
                            
                            # Analyze with DeepFace
                            result = DeepFace.analyze(
                                face_resized,
                                actions=['age'],
                                enforce_detection=False,
                                silent=True
                            )
                            
                            # Extract age
                            if isinstance(result, list):
                                deepface_age = result[0]['age']
                            else:
                                deepface_age = result['age']
                            
                            # Clamp to reasonable range
                            deepface_age = max(10, min(100, deepface_age))
                            
                            # Clean up memory
                            del face_resized, result
                            gc.collect()
                            
                        else:
                            st.warning(f"Face {i+1} too small to analyze")
                            
                    except Exception as e:
                        st.error(f"❌ DeepFace failed for face {i+1}: {str(e)}")
                    
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        face_pil = Image.fromarray(face_crop)
                        st.image(face_pil, caption=f"Face {i+1}", width=150)
                    
                    with col2:
                        # Show both ages if available
                        if harvard_age is not None:
                            st.metric("🎯 Harvard Age", f"{harvard_age:.1f} years")
                        if deepface_age is not None:
                            st.metric("🤖 DeepFace Age", f"{deepface_age:.1f} years")
                        
                        st.metric("Detection Confidence", f"{face['confidence']:.2f}")
                        
                        # Use Harvard age as primary, DeepFace as fallback
                        primary_age = harvard_age if harvard_age is not None else deepface_age
                        
                        if primary_age is not None:
                            if primary_age < 30:
                                st.success("😊 Young")
                            elif primary_age < 50:
                                st.info("😐 Adult")
                            else:
                                st.warning("👴 Senior")
                        else:
                            st.error("❓ Could not determine age")
                    
                    st.markdown("---")
                
                # Clean up memory
                del img_array, faces
                gc.collect()
                
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.error("Please try a different image or refresh the page.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🔬 AI-Powered Age Estimation • 🌟 Cloud Optimized • �� Harvard + DeepFace</p>
</div>
""", unsafe_allow_html=True) 