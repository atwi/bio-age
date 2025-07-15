import streamlit as st
import sys
import os

st.title("🔍 Health Check - Facial Age Estimator")
st.write("This page helps diagnose deployment issues")

# Environment info
st.subheader("🌍 Environment Information")
st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")

# Check if running on cloud
if 'STREAMLIT_CLOUD' in os.environ:
    st.success("✅ Running on Streamlit Cloud")
elif 'HOSTNAME' in os.environ:
    st.success(f"✅ Running on: {os.environ['HOSTNAME']}")
else:
    st.info("ℹ️ Running locally")

# Test imports
st.subheader("📦 Package Imports")

packages = [
    ('numpy', 'import numpy as np'),
    ('opencv', 'import cv2'),
    ('PIL', 'from PIL import Image'),
    ('matplotlib', 'import matplotlib'),
    ('mtcnn', 'from mtcnn import MTCNN'),
    ('tensorflow', 'import tensorflow as tf'),
    ('keras', 'import keras'),
    ('deepface', 'from deepface import DeepFace'),
    ('gdown', 'import gdown'),
]

for name, import_cmd in packages:
    try:
        exec(import_cmd)
        st.success(f"✅ {name}")
    except Exception as e:
        st.error(f"❌ {name}: {e}")

# Test basic functionality
st.subheader("🧪 Functionality Tests")

try:
    import numpy as np
    import cv2
    from mtcnn import MTCNN
    import tensorflow as tf
    import keras
    from deepface import DeepFace
    
    # Version check
    st.write(f"TensorFlow: {tf.__version__}")
    st.write(f"Keras: {keras.__version__}")
    st.write(f"OpenCV: {cv2.__version__}")
    
    # Test face detection
    detector = MTCNN()
    st.success("✅ MTCNN face detection initialized")
    
    # Test DeepFace
    test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = DeepFace.analyze(test_face, actions=['age'], enforce_detection=False)
    st.success("✅ DeepFace age estimation working")
    
    st.balloons()
    st.success("🎉 All systems operational!")
    
except Exception as e:
    st.error(f"❌ Test failed: {e}")
    import traceback
    st.code(traceback.format_exc())

st.markdown("---")
st.write("If all tests pass, the main app should work correctly.")
st.write("If any test fails, please check the error details above.") 