# DEPRECATION NOTICE
# This module is kept for legacy reference only and is not used by the current app.
# Use `main.py`, which exposes routes under the `/api` prefix (e.g., `/api/analyze-face`).
# Differences: this file uses `/analyze-face` without `/api` and has simplified loading.
print("\u26a0\ufe0f DEPRECATED: 'api_backend.py' is legacy. Use 'main.py' (with '/api' routes).", flush=True)

import os
import sys
import gc
import warnings
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from mtcnn import MTCNN
import tensorflow as tf
import tensorflow.keras as keras
import zipfile
import gdown
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
import json

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cloud environment detection
IS_CLOUD = (
    'STREAMLIT_CLOUD' in os.environ or 
    'HOSTNAME' in os.environ or 
    'DYNO' in os.environ or
    'RAILWAY_ENVIRONMENT' in os.environ
)

if IS_CLOUD:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Initialize FastAPI
app = FastAPI(
    title="TrueAge API",
    description="TrueAge uses advanced AI models to estimate your biological age and perceived age from a single facial photo. Upload your selfie to discover insights about your health and aging, powered by state-of-the-art deep learning and facial analysis technology.",
    version="1.0.0"
)

# Enable CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
face_detector = None
harvard_model = None

# Model configuration
MODEL_ZIP = "model_saved_tf.zip"
MODEL_DIR = "model_saved_tf"
GDRIVE_FILE_ID = "12wNpYBz3j5mP9mt6S_ZH4k0sI6dVDeVV"

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if IS_CLOUD:
        try:
            tf.keras.backend.clear_session()
        except:
            pass

def init_face_detector():
    """Initialize MTCNN face detector"""
    global face_detector
    if face_detector is None:
        try:
            face_detector = MTCNN()
            return True
        except Exception as e:
            print(f"Face detector initialization failed: {e}")
            return False
    return True

def download_harvard_model():
    """Download Harvard model if not exists"""
    if not os.path.exists(MODEL_DIR):
        try:
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            print("Downloading Harvard model...")
            gdown.download(url, MODEL_ZIP, quiet=True)
            
            with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
                zip_ref.extractall(".")
            
            if os.path.exists(MODEL_ZIP):
                os.remove(MODEL_ZIP)
            
            return True
        except Exception as e:
            print(f"Harvard model download failed: {e}")
            return False
    return True

def load_harvard_model():
    """Load Harvard model"""
    global harvard_model
    if harvard_model is not None:
        return True
        
    if not download_harvard_model():
        return False
    
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
                    harvard_model = keras.models.load_model(model_path)
                else:
                    try:
                        harvard_model = tf.keras.models.load_model(model_path)
                    except Exception:
                        harvard_model = tf.saved_model.load(model_path)
                
                print(f"Harvard model loaded from: {model_path}")
                return True
            except Exception as e:
                print(f"Failed to load from {model_path}: {e}")
                continue
    
    print("Harvard model not available")
    return False

def test_deepface():
    """Test DeepFace functionality"""
    try:
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = DeepFace.analyze(
            test_img,
            actions=['age'],
            enforce_detection=False,
            silent=True
        )
        print("DeepFace test successful")
        return True
    except Exception as e:
        print(f"DeepFace test failed: {e}")
        return False

def predict_age_harvard(face_crop):
    """Predict age using Harvard model"""
    if harvard_model is None:
        return None
    
    try:
        # Prepare face for Harvard model
        face_resized = cv2.resize(face_crop, (160, 160))
        face_pil = Image.fromarray(face_resized).convert('RGB')
        face_array = np.asarray(face_pil)
        mean, std = face_array.mean(), face_array.std()
        if std == 0:
            std = 1
        face_normalized = (face_array - mean) / std
        face_input = face_normalized.reshape(1, 160, 160, 3)
        
        # Predict age
        if hasattr(harvard_model, 'predict'):
            prediction = harvard_model.predict(face_input, verbose=0)
            age = float(np.squeeze(prediction))
        elif hasattr(harvard_model, 'signatures'):
            signature = harvard_model.signatures['serving_default']
            prediction = signature(tf.constant(face_input, dtype=tf.float32))
            if isinstance(prediction, dict):
                age_prediction = list(prediction.values())[0]
            else:
                age_prediction = prediction
            age = float(np.squeeze(age_prediction))
        else:
            prediction = harvard_model(face_input)
            if isinstance(prediction, dict):
                age_prediction = list(prediction.values())[0]
            else:
                age_prediction = prediction
            age = float(np.squeeze(age_prediction))
        
        # Clamp age to reasonable range
        age = max(0, min(120, age))
        
        # Clean up
        del face_input, prediction
        if 'age_prediction' in locals():
            del age_prediction
        
        return age
        
    except Exception as e:
        print(f"Harvard model prediction failed: {e}")
        return None

def predict_age_deepface(face_crop):
    """Predict age using DeepFace"""
    try:
        if face_crop.shape[0] <= 20 or face_crop.shape[1] <= 20:
            return None
            
        face_resized = cv2.resize(face_crop, (224, 224))
        face_resized = np.clip(face_resized, 0, 255).astype(np.uint8)
        
        result = DeepFace.analyze(
            face_resized,
            actions=['age'],
            enforce_detection=False,
            silent=True
        )
        
        # Extract age
        if isinstance(result, list):
            age = result[0]['age']
        else:
            age = result['age']
        
        # Clamp to reasonable range
        age = max(10, min(100, age))
        
        # Clean up
        del face_resized, result
        aggressive_cleanup()
        
        return age
        
    except Exception as e:
        print(f"DeepFace prediction failed: {e}")
        return None

def create_face_overlay(image_array, faces):
    """Create face detection overlay"""
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    
    # Try to load font
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
        small_font = ImageFont.truetype("Arial.ttf", 18)
    except:
        font = None
        small_font = None
    
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        confidence = face['confidence']
        
        # Colors for tech look
        primary_color = "#00FF00"  # Bright green
        secondary_color = "#00CC00"  # Darker green
        bg_color = "#000000"  # Black background
        
        # Draw corner brackets
        bracket_size = min(w, h) // 6
        line_width = 5
        
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
        
        # Badge position
        badge_x = x + 5
        badge_y = y - 35
        badge_padding = 8
        
        # Draw badge background
        badge_bg = [
            badge_x - badge_padding,
            badge_y - badge_padding,
            badge_x + text_width + badge_padding,
            badge_y + text_height + badge_padding
        ]
        
        draw.rectangle(badge_bg, fill=bg_color + "E0", outline=primary_color, width=3)
        draw.text((badge_x, badge_y), badge_text, fill=primary_color, font=small_font)
        
        # Add face number
        face_num_text = f"{i+1}"
        if font:
            num_bbox = draw.textbbox((0, 0), face_num_text, font=font)
            num_width = num_bbox[2] - num_bbox[0]
            num_height = num_bbox[3] - num_bbox[1]
        else:
            num_width = 16
            num_height = 16
        
        # Draw face number in bottom-right
        num_x = x + w - num_width - 10
        num_y = y + h - num_height - 10
        
        # Circle background
        circle_radius = 18
        draw.ellipse(
            [num_x - circle_radius, num_y - circle_radius, 
             num_x + circle_radius, num_y + circle_radius],
            fill=secondary_color, outline=primary_color, width=3
        )
        
        # Draw number
        draw.text((num_x - num_width//2, num_y - num_height//2), 
                 face_num_text, fill="white", font=font)
    
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Initializing models...")
    
    if not init_face_detector():
        print("Warning: Face detector initialization failed")
    
    if not load_harvard_model():
        print("Warning: Harvard model not available")
    
    if not test_deepface():
        print("Warning: DeepFace not available")
    
    print("API server ready!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models": {
        "face_detector": face_detector is not None,
        "harvard_model": harvard_model is not None,
        "deepface": True  # Always available
    }}

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """Analyze faces in uploaded image"""
    try:
        # Check if models are loaded
        if face_detector is None:
            raise HTTPException(status_code=500, detail="Face detector not initialized")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)
        
        # Detect faces
        faces = face_detector.detect_faces(img_array)
        
        if not faces:
            return JSONResponse({
                "success": False,
                "message": "No faces detected",
                "faces": []
            })
        
        # Create detection overlay
        overlay_image = create_face_overlay(img_array, faces)
        overlay_base64 = image_to_base64(overlay_image)
        
        # Analyze each face
        results = []
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            
            # Extract face crop
            face_crop = img_array[y:y+h, x:x+w]
            
            # Predict ages
            harvard_age = predict_age_harvard(face_crop)
            deepface_age = predict_age_deepface(face_crop)
            
            # Create face crop image
            face_pil = Image.fromarray(face_crop)
            face_base64 = image_to_base64(face_pil)
            
            # Determine primary age and category
            primary_age = harvard_age if harvard_age is not None else deepface_age
            
            if primary_age is not None:
                if primary_age < 30:
                    category = "young"
                elif primary_age < 50:
                    category = "adult"
                else:
                    category = "senior"
            else:
                category = "unknown"
            
            results.append({
                "face_id": i + 1,
                "bounding_box": face['box'],
                "confidence": face['confidence'],
                "harvard_age": harvard_age,
                "deepface_age": deepface_age,
                "primary_age": primary_age,
                "category": category,
                "face_crop_base64": face_base64,
                "warnings": []
            })
            
            # Add warnings for DeepFace bias
            if deepface_age is not None and 35 <= deepface_age <= 55:
                results[-1]["warnings"].append("DeepFace prediction may be affected by training bias toward middle ages")
        
        # Clean up
        del img_array, faces
        aggressive_cleanup()
        
        return JSONResponse({
            "success": True,
            "message": f"Found {len(results)} face(s)",
            "faces": results,
            "overlay_image_base64": overlay_base64,
            "image_dimensions": {
                "width": image.width,
                "height": image.height
            }
        })
        
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Analysis failed: {str(e)}"}
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "TrueAge API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 