import streamlit as st
import requests
import json
from PIL import Image
import io
import base64

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Facial Age Estimator (API Client)",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 0.5rem; }
    .stButton > button { width: 100%; height: 3rem; border-radius: 10px; }
    @media (max-width: 768px) { .main { padding: 0.25rem; } }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📱 Facial Age Estimator (API Client)")
st.markdown("**Estimate biological age from facial features using API backend**")

# Check API health
def check_api_health():
    """Check if the API backend is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        else:
            return False, {"error": f"API returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def analyze_image_via_api(image_file):
    """Send image to API for analysis"""
    try:
        # Prepare file for upload
        files = {"file": ("image.jpg", image_file, "image/jpeg")}
        
        # Send request to API
        response = requests.post(f"{API_BASE_URL}/analyze-face", files=files, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        st.error(f"Error converting base64 to image: {e}")
        return None

# Check API status
st.markdown("### 🔌 API Status")
api_healthy, health_data = check_api_health()

if api_healthy:
    st.success("✅ API backend is healthy")
    
    # Show model status
    models = health_data.get("models", {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detector_status = "✅" if models.get("face_detector") else "❌"
        st.metric("Face Detector", detector_status)
    
    with col2:
        harvard_status = "✅" if models.get("harvard_model") else "❌"
        st.metric("Harvard Model", harvard_status)
    
    with col3:
        deepface_status = "✅" if models.get("deepface") else "❌"
        st.metric("DeepFace", deepface_status)
else:
    st.error("❌ API backend is not available")
    st.error(f"Error: {health_data.get('error', 'Unknown error')}")
    st.info("Please make sure the API backend is running on port 8000")
    st.code("python3 api_backend.py")
    st.stop()

st.markdown("---")

# Main interface
st.markdown("### 📸 Upload Your Photo")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo with visible face(s)"
)

if uploaded_file is not None:
    with st.spinner("🔍 Analyzing faces via API..."):
        # Convert uploaded file to bytes
        image_bytes = uploaded_file.getvalue()
        
        # Send to API
        success, result = analyze_image_via_api(io.BytesIO(image_bytes))
        
        if success and result.get("success"):
            st.success(result.get("message", "Analysis completed"))
            
            # Show overlay image
            overlay_base64 = result.get("overlay_image_base64")
            if overlay_base64:
                overlay_image = base64_to_image(overlay_base64)
                if overlay_image:
                    st.image(overlay_image, caption="🎯 Face Detection Results", width=400)
            
            # Show individual face results
            faces = result.get("faces", [])
            
            for face in faces:
                st.markdown(f"### 👤 Face {face.get('face_id', 'N/A')}")
                
                # Display face crop and results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    face_crop_base64 = face.get("face_crop_base64")
                    if face_crop_base64:
                        face_crop = base64_to_image(face_crop_base64)
                        if face_crop:
                            st.image(face_crop, caption=f"Face {face.get('face_id', 'N/A')} (Cropped)", width=150)
                
                with col2:
                    # Show age predictions
                    harvard_age = face.get("harvard_age")
                    deepface_age = face.get("deepface_age")
                    
                    if harvard_age is not None:
                        st.metric("🎯 Harvard Age", f"{harvard_age:.1f} years")
                    if deepface_age is not None:
                        st.metric("🤖 DeepFace Age", f"{deepface_age:.1f} years")
                    
                    # Show detection confidence
                    confidence = face.get("confidence")
                    if confidence is not None:
                        st.metric("Face Detection (MTCNN)", f"{confidence:.2f}")
                    
                    # Show category
                    category = face.get("category", "unknown")
                    primary_age = face.get("primary_age")
                    
                    if primary_age is not None:
                        if category == "young":
                            st.success("😊 Young")
                        elif category == "adult":
                            st.info("😐 Adult")
                        elif category == "senior":
                            st.warning("👴 Senior")
                        else:
                            st.error("❓ Could not determine age")
                    
                    # Show warnings
                    warnings = face.get("warnings", [])
                    for warning in warnings:
                        st.warning(f"⚠️ {warning}")
                
                st.markdown("---")
        
        elif success and not result.get("success"):
            st.error(result.get("message", "No faces detected"))
        else:
            st.error("❌ Failed to analyze image")
            st.error(result.get("error", "Unknown error"))

# Model information
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🔬 AI-Powered Age Estimation • 🌟 API Backend • 🤖 DeepFace + Harvard</p>
</div>
""", unsafe_allow_html=True) 