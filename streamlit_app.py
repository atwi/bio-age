import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
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

# Memory management function
def create_face_overlay(image, faces):
    """Create a clean tech-style overlay showing face detection"""
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
        small_font = ImageFont.truetype("Arial.ttf", 18)
    except:
        try:
            # Try system default
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        except:
            font = None
            small_font = None
    
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        confidence = face['confidence']
        
        # Colors for tech look - make more prominent
        primary_color = "#00FF00"  # Bright green for better visibility
        secondary_color = "#00CC00"  # Darker green
        bg_color = "#000000"  # Pure black background
        
        # Draw corner brackets (clean tech style) - make more prominent
        bracket_size = min(w, h) // 6  # Larger brackets
        line_width = 5  # Thicker lines
        
        # Top-left corner
        draw.line([(x, y), (x + bracket_size, y)], fill=primary_color, width=line_width)
        draw.line([(x, y), (x, y + bracket_size)], fill=primary_color, width=line_width)
        
        # Top-right corner
        draw.line([(x + w - bracket_size, y), (x + w, y)], fill=primary_color, width=line_width)
        draw.line([(x + w, y), (x + w, y + bracket_size)], fill=primary_color, width=line_width)
        
        # Bottom-left corner
        draw.line([(x, y + h - bracket_size), (x, y + h)], fill=primary_color, width=line_width)
        draw.line([(x, y + h), (x + bracket_size, y + h)], fill=primary_color, width=line_width)
        
        # Bottom-right corner
        draw.line([(x + w - bracket_size, y + h), (x + w, y + h)], fill=primary_color, width=line_width)
        draw.line([(x + w, y + h - bracket_size), (x + w, y + h)], fill=primary_color, width=line_width)
        
        # Create confidence badge
        badge_text = f"Face {i+1}: {confidence:.1%}"
        
        # Calculate badge dimensions
        if font:
            bbox = draw.textbbox((0, 0), badge_text, font=small_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(badge_text) * 8
            text_height = 16
        
        # Badge position (top-left, slightly offset)
        badge_x = x + 5
        badge_y = y - 35
        badge_padding = 8
        
        # Draw badge background with rounded corners effect
        badge_bg = [
            badge_x - badge_padding,
            badge_y - badge_padding,
            badge_x + text_width + badge_padding,
            badge_y + text_height + badge_padding
        ]
        
        # Semi-transparent background with better visibility
        draw.rectangle(badge_bg, fill=bg_color + "E0", outline=primary_color, width=3)
        
        # Draw badge text
        draw.text((badge_x, badge_y), badge_text, fill=primary_color, font=small_font)
        
        # Add small face number inside the detection box
        face_num_text = f"{i+1}"
        if font:
            num_bbox = draw.textbbox((0, 0), face_num_text, font=font)
            num_width = num_bbox[2] - num_bbox[0]
            num_height = num_bbox[3] - num_bbox[1]
        else:
            num_width = 16
            num_height = 16
        
        # Draw face number in bottom-right of detection box
        num_x = x + w - num_width - 10
        num_y = y + h - num_height - 10
        
        # Small circle background for number - more prominent
        circle_radius = 18
        draw.ellipse(
            [num_x - circle_radius, num_y - circle_radius, 
             num_x + circle_radius, num_y + circle_radius],
            fill=secondary_color, outline=primary_color, width=3
        )
        
        # Draw number
        draw.text((num_x - num_width//2, num_y - num_height//2), 
                 face_num_text, fill="white", font=font)
    
    return pil_image

def aggressive_cleanup():
    """Aggressive memory cleanup for cloud environments"""
    gc.collect()
    if IS_CLOUD:
        try:
            # Clear TensorFlow backend session
            tf.keras.backend.clear_session()
        except:
            pass
    
# Page configuration for mobile
st.set_page_config(
    page_title="Facial Age Estimator",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cloud environment info with memory monitoring
if IS_CLOUD:
    st.info("🌟 Running on Streamlit Cloud with Memory Optimization")
    
    # Memory monitoring for cloud
    try:
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        st.caption(f"Memory: {memory_percent}% used")
        
        # Warning if memory is high
        if memory_percent > 80:
            st.warning("⚠️ High memory usage detected - optimizing...")
            aggressive_cleanup()
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

# Main interface - moved up for better UX
st.markdown("### 📸 Upload Your Photo")

# Simplified upload (cloud-friendly)
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo with visible face(s)"
)

# Model information and credits - moved below upload
st.markdown("---")
st.markdown("### 🎯 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🏫 Harvard FaceAge Model**
    - **Specifically designed for ages 60+**
    - More accurate for biological aging
    - Trained on clinical populations
    - **Recommended for elderly subjects**
    """)

with col2:
    st.markdown("""
    **🤖 DeepFace Model**
    - General-purpose face analysis
    - **⚠️ Training bias toward middle ages**
    - **Poor accuracy for elderly faces (70+)**
    - Tends to predict 40-50 for all ages
    """)

st.warning("⚠️ **Important**: DeepFace has known limitations for elderly faces due to training bias. It often predicts middle-aged values (40-50) even for obviously elderly subjects. The Harvard model is specifically designed for better accuracy on older faces.")

st.info("🔍 **Technical Note**: The app uses MTCNN for face detection, then feeds the detected faces to both models for age estimation. DeepFace's internal face detection is disabled to avoid conflicts.")

st.markdown("### 📚 Model Credits")
st.markdown("""
- [**Harvard FaceAge**](github.com/AIM-Harvard/FaceAge) - Research model for biological age estimation
- [**DeepFace**](https://github.com/serengil/deepface) - Comprehensive face analysis framework
- **MTCNN**: Face detection and alignment
""")

st.markdown("---")

# Cloud-optimized model handling
MODEL_ZIP = "model_saved_tf.zip"
MODEL_DIR = "model_saved_tf"
GDRIVE_FILE_ID = "12wNpYBz3j5mP9mt6S_ZH4k0sI6dVDeVV"

# Remove caching for large models to prevent memory issues
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

# Remove caching to prevent memory buildup
def load_harvard_model():
    """Load Harvard model with enhanced cloud compatibility - NO CACHING"""
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

# Remove caching for DeepFace test to prevent memory buildup
def test_deepface_cloud():
    """Test DeepFace with cloud-specific settings - NO CACHING"""
    try:
        # Create minimal test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        st.write("🔍 **DeepFace Initialization Test:**")
        st.write(f"- Test image shape: {test_img.shape}")
        
        # Cloud-optimized DeepFace call
        result = DeepFace.analyze(
            test_img,
            actions=['age'],
            enforce_detection=False,
            silent=True
        )
        
        st.write(f"- Test result: {result}")
        
        # Extract test age
        if isinstance(result, list):
            test_age = result[0]['age']
            st.write(f"- Test age from list: {test_age}")
        else:
            test_age = result['age']
            st.write(f"- Test age from dict: {test_age}")
        
        # Force garbage collection
        del test_img, result
        aggressive_cleanup()
        
        st.success(f"✅ DeepFace initialization successful! Test age: {test_age}")
        return True
        
    except Exception as e:
        st.error(f"❌ DeepFace test failed: {str(e)}")
        st.write(f"🔍 **DeepFace Init Error:** {type(e).__name__}: {str(e)}")
        return False

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.detector = None

def lazy_load_models():
    """Lazy load models only when user uploads an image - NO PERSISTENT CACHING"""
    if st.session_state.models_loaded:
        return True
    
    with st.spinner("🔄 Loading AI models... This may take a moment."):
        try:
            # Step 1: Face detector (keep cached as it's small)
            st.session_state.detector = init_face_detector()
            if st.session_state.detector is None:
                st.error("❌ Face detection failed - app cannot continue")
                return False
            
            # Step 2: Test DeepFace (no caching)
            deepface_ok = test_deepface_cloud()
            if not deepface_ok:
                st.error("❌ DeepFace failed - app cannot continue")
                return False
            
            st.session_state.models_loaded = True
            st.success("✅ Ready! Models initialized")
            
            # Memory cleanup
            aggressive_cleanup()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            return False

if uploaded_file is not None:
    # Lazy load models only when user uploads an image
    if not lazy_load_models():
        st.error("❌ Failed to load models. Please refresh the page.")
        st.stop()
    
    # Process image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Process with cloud optimizations
        with st.spinner("🔍 Analyzing faces..."):
            img_array = np.array(image)
            
            # Face detection
            faces = st.session_state.detector.detect_faces(img_array)
            
            if not faces:
                st.error("❌ No face detected. Please try a clearer photo.")
                # Show original image only if no faces found
                st.image(image, caption="📸 Original photo - No faces detected", width=400)
            else:
                st.success(f"✅ Found {len(faces)} face(s)")
                
                # Create and display face detection overlay (replaces original image)
                overlay_image = create_face_overlay(image, faces)
                st.image(overlay_image, caption="🎯 Face Detection Results", width=400)
                
                # Load Harvard model fresh each time (no caching)
                harvard_model = None
                # Re-enable Harvard model in cloud since it's smaller than DeepFace (85MB vs 539MB)
                harvard_model = load_harvard_model()
                
                # Process each face
                for i, face in enumerate(faces):
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    
                    # Extract face
                    face_crop = img_array[y:y+h, x:x+w]
                    
                    # Try Harvard model first (local only)
                    harvard_age = None
                    if harvard_model is not None:
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
                                if hasattr(harvard_model, 'predict'):
                                    # Standard Keras model
                                    prediction = harvard_model.predict(face_input, verbose=0)
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
                                
                                # Clamp age to reasonable range
                                harvard_age = max(0, min(120, harvard_age))
                                
                                # Clean up prediction variables
                                del face_input, prediction
                                if 'age_prediction' in locals():
                                    del age_prediction
                                
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
                            # Note: DeepFace has training bias toward middle ages (40-50)
                            # It's not optimal for elderly faces due to dataset bias
                            face_resized = cv2.resize(face_crop, (224, 224))
                            face_resized = np.clip(face_resized, 0, 255).astype(np.uint8)
                            
                            # Analyze with DeepFace (using MTCNN's face detection results)
                            result = DeepFace.analyze(
                                face_resized,
                                actions=['age'],
                                enforce_detection=False,  # Don't fail if DeepFace disagrees with MTCNN
                                silent=True
                            )
                            
                            # Extract age (ignore DeepFace's internal face confidence - trust MTCNN)
                            if isinstance(result, list):
                                deepface_age = result[0]['age']
                            else:
                                deepface_age = result['age']
                            
                            # Warning for training bias (main issue with DeepFace)
                            if deepface_age >= 35 and deepface_age <= 55:
                                st.warning("⚠️ DeepFace prediction may be affected by training bias toward middle ages")
                            
                            # Clamp to reasonable range
                            deepface_age = max(10, min(100, deepface_age))
                            
                            # Clean up memory aggressively
                            del face_resized, result
                            aggressive_cleanup()
                            
                        else:
                            st.warning(f"Face {i+1} too small to analyze")
                            
                    except Exception as e:
                        st.error(f"❌ DeepFace failed for face {i+1}: {str(e)}")
                        st.write(f"🔍 **DeepFace Error Details:** {type(e).__name__}: {str(e)}")
                    
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        face_pil = Image.fromarray(face_crop)
                        st.image(face_pil, caption=f"Face {i+1} (Cropped)", width=150)
                    
                    with col2:
                        # Show both ages if available
                        if harvard_age is not None:
                            st.metric("🎯 Harvard Age", f"{harvard_age:.1f} years")
                        if deepface_age is not None:
                            st.metric("🤖 DeepFace Age", f"{deepface_age:.1f} years")
                        
                        st.metric("Face Detection (MTCNN)", f"{face['confidence']:.2f}")
                        
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
                
                # Clean up harvard model to free memory
                if harvard_model is not None:
                    del harvard_model
                    
                # Clean up processing variables
                del img_array, faces
                aggressive_cleanup()
                
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.error("Please try a different image or refresh the page.")
        # Clean up on error
        aggressive_cleanup()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🔬 AI-Powered Age Estimation • 🌟 Memory Optimized • 🤖 DeepFace + Harvard</p>
</div>
""", unsafe_allow_html=True) 