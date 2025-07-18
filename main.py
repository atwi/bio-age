#!/usr/bin/env python3
"""
Main deployment script for Railway/Render
Serves both the FastAPI backend and React Native web frontend
Railway-optimized with graceful fallback and memory optimization
"""

import os
import sys
import gc
import time
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import openai
from dotenv import load_dotenv
import os

# Global variables for lazy loading
detector = None
harvard_model = None
deepface_ready = False
models_loading = False

# Load environment variables
load_dotenv()

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# Environment detection
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') == 'production'
LOAD_HARVARD = os.environ.get('LOAD_HARVARD_MODEL', 'true').lower() == 'true'
ENABLE_DEEPFACE = os.environ.get('ENABLE_DEEPFACE', 'true').lower() == 'true'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if IS_RAILWAY:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class FaceResult(BaseModel):
    face_id: int
    age_harvard: Optional[float] = None
    age_deepface: Optional[float] = None
    age_chatgpt: Optional[int] = None
    confidence: float
    face_crop_base64: str

class AnalyzeResponse(BaseModel):
    success: bool
    faces: List[FaceResult]
    message: str

def lazy_load_models():
    """Lazy load models only when needed"""
    global detector, harvard_model, deepface_ready, models_loading
    
    if models_loading:
        return False
    
    if detector is not None and (harvard_model is not None or not LOAD_HARVARD) and (deepface_ready or not ENABLE_DEEPFACE):
        return True
    
    models_loading = True
    logger.info("🔄 Lazy loading models...")
    
    try:
        # Load face detector first (smallest)
        if detector is None:
            logger.info("Loading face detector...")
            from mtcnn import MTCNN
            detector = MTCNN()
            logger.info("✅ Face detector loaded")
            gc.collect()
        
        # Load Harvard model if enabled
        if LOAD_HARVARD and harvard_model is None:
            logger.info("Loading Harvard model...")
            harvard_model = load_harvard_model()
            if harvard_model:
                logger.info("✅ Harvard model loaded")
            gc.collect()
        
        # Test DeepFace last (largest) - only if enabled
        if ENABLE_DEEPFACE and not deepface_ready:
            logger.info("Initializing DeepFace...")
            deepface_ready = test_deepface()
            if deepface_ready:
                logger.info("✅ DeepFace initialized")
            gc.collect()
        elif not ENABLE_DEEPFACE:
            logger.info("🚫 DeepFace disabled by configuration")
            deepface_ready = True  # Mark as ready to skip loading
        
        models_loading = False
        return detector is not None and (harvard_model is not None or not LOAD_HARVARD) and (deepface_ready or not ENABLE_DEEPFACE)
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        models_loading = False
        return False

def load_harvard_model():
    """Load Harvard model with enhanced error handling"""
    import tensorflow as tf
    import keras
    
    # Try different possible paths
    possible_paths = [
        "FaceAge/models/model_saved_tf",
        "./FaceAge/models/model_saved_tf",
        "model_saved_tf",
        "./model_saved_tf"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading Harvard model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                logger.warning(f"Failed to load from {model_path}: {e}")
                continue
    
    logger.warning("Harvard model not found or failed to load")
    return None

def test_deepface():
    """Test DeepFace initialization"""
    try:
        from deepface import DeepFace
        
        # Create minimal test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test DeepFace
        result = DeepFace.analyze(
            test_img,
            actions=['age'],
            enforce_detection=False,
            silent=True
        )
        
        # Clean up
        del test_img, result
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"DeepFace test failed: {e}")
        return False

def estimate_age_chatgpt(image_base64: str) -> Optional[int]:
    """Estimate age using ChatGPT Vision with function calling"""
    try:
        # Function definition for structured response
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "estimate_age",
                    "description": "Estimate the age of the person in the image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "age_years": {
                                "type": "integer",
                                "description": "Estimated age in years (must be between 18-100)",
                                "minimum": 18,
                                "maximum": 100
                            }
                        },
                        "required": ["age_years"]
                    }
                }
            }
        ]
        
        # Prepare the image for ChatGPT
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Call ChatGPT Vision API
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Look at this face and estimate the person's age. Consider facial features, skin texture, wrinkles, and overall appearance. Respond with ONLY the estimated age in years."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            tools=functions,
            tool_choice={"type": "function", "function": {"name": "estimate_age"}},
            max_tokens=50
        )
        
        # Extract the function call result
        if response.choices[0].message.tool_calls:
            import json
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            age = function_args.get("age_years")
            if age and 18 <= age <= 100:
                return age
        
        # Fallback: try to parse the response text
        response_text = response.choices[0].message.content
        import re
        age_match = re.search(r'\b(\d{2,3})\b', response_text)
        if age_match:
            age = int(age_match.group(1))
            if 18 <= age <= 100:
                return age
        
        return None
        
    except Exception as e:
        logger.error(f"ChatGPT age estimation failed: {e}")
        return None

# FastAPI app without startup model loading
app = FastAPI(
    title="Bio Age Estimator API",
    description="Facial age estimation using Harvard FaceAge and DeepFace models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint serving the React Native web build"""
    web_build_path = "FaceAgeApp/dist"
    if os.path.exists(f"{web_build_path}/index.html"):
        return FileResponse(f"{web_build_path}/index.html")
    return {"message": "Bio Age Estimator API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/api/health")
async def api_health_check():
    """API health check with model status"""
    global detector, harvard_model, deepface_ready
    
    return {
        "status": "healthy",
        "models": {
            "face_detector": detector is not None,
            "harvard_model": harvard_model is not None,
            "deepface": deepface_ready if ENABLE_DEEPFACE else "disabled"
        },
        "config": {
            "enable_deepface": ENABLE_DEEPFACE,
            "load_harvard": LOAD_HARVARD,
            "is_railway": IS_RAILWAY
        },
        "timestamp": time.time()
    }

@app.post("/api/analyze-face", response_model=AnalyzeResponse)
async def analyze_face(file: UploadFile = File(...)):
    """Analyze face with lazy model loading"""
    try:
        # Lazy load models on first request
        if not lazy_load_models():
            raise HTTPException(status_code=500, detail="Models failed to load")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        img_array = np.array(image)
        
        # Detect faces
        faces = detector.detect_faces(img_array)
        
        if not faces:
            return AnalyzeResponse(
                success=False,
                faces=[],
                message="No face detected"
            )
        
        # Filter faces with confidence >= 0.9
        high_confidence_faces = [f for f in faces if f['confidence'] >= 0.9]
        
        if not high_confidence_faces:
            return AnalyzeResponse(
                success=False,
                faces=[],
                message="No high-confidence faces detected (≥90% confidence required)"
            )
        
        results = []
        
        for i, face in enumerate(high_confidence_faces):
            try:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                
                # Extract face crop
                face_crop = img_array[y:y+h, x:x+w]
                
                # Create base64 face crop
                face_pil = Image.fromarray(face_crop)
                face_buffer = BytesIO()
                face_pil.save(face_buffer, format='PNG')
                face_base64 = base64.b64encode(face_buffer.getvalue()).decode('utf-8')
                
                # Harvard model prediction
                harvard_age = None
                if harvard_model is not None:
                    try:
                        import cv2
                        face_resized = cv2.resize(face_crop, (160, 160))
                        face_pil_resized = Image.fromarray(face_resized).convert('RGB')
                        face_array = np.asarray(face_pil_resized)
                        mean, std = face_array.mean(), face_array.std()
                        if std == 0:
                            std = 1
                        face_normalized = (face_array - mean) / std
                        face_input = face_normalized.reshape(1, 160, 160, 3)
                        
                        prediction = harvard_model.predict(face_input, verbose=0)
                        harvard_age = float(np.squeeze(prediction))
                        harvard_age = max(0, min(120, harvard_age))
                        
                    except Exception as e:
                        logger.warning(f"Harvard prediction failed: {e}")
                
                # DeepFace prediction
                deepface_age = None
                if ENABLE_DEEPFACE and deepface_ready:
                    try:
                        import cv2
                        from deepface import DeepFace
                        
                        face_resized = cv2.resize(face_crop, (224, 224))
                        face_resized = np.clip(face_resized, 0, 255).astype(np.uint8)
                        
                        result = DeepFace.analyze(
                            face_resized,
                            actions=['age'],
                            enforce_detection=False,
                            silent=True
                        )
                        
                        if isinstance(result, list):
                            deepface_age = result[0]['age']
                        else:
                            deepface_age = result['age']
                        
                        deepface_age = max(10, min(100, deepface_age))
                        
                    except Exception as e:
                        logger.warning(f"DeepFace prediction failed: {e}")
                elif not ENABLE_DEEPFACE:
                    logger.info("🚫 DeepFace prediction skipped (disabled)")
                
                # ChatGPT prediction
                chatgpt_age = None
                try:
                    chatgpt_age = estimate_age_chatgpt(face_base64)
                except Exception as e:
                    logger.warning(f"ChatGPT prediction failed: {e}")
                
                results.append(FaceResult(
                    face_id=i,
                    age_harvard=harvard_age,
                    age_deepface=deepface_age,
                    age_chatgpt=chatgpt_age,
                    confidence=face['confidence'],
                    face_crop_base64=face_base64
                ))
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        return AnalyzeResponse(
            success=True,
            faces=results,
            message=f"Found {len(results)} high-confidence face(s)"
        )
        
    except Exception as e:
        logger.error(f"Error analyzing face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for React Native web build
web_build_path = "FaceAgeApp/dist"
if os.path.exists(web_build_path):
    app.mount("/", StaticFiles(directory=web_build_path, html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting Bio Age Estimator on port {port}")
    print(f"📁 Web build path: {web_build_path}")
    print(f"🌐 Frontend available: {os.path.exists(web_build_path)}")
    print(f"🔧 Railway environment: {IS_RAILWAY}")
    print(f"📊 Harvard model enabled: {LOAD_HARVARD}")
    print(f"🤖 DeepFace enabled: {ENABLE_DEEPFACE}")
    print(f"⚡ Using lazy loading for models")
    
    uvicorn.run(app, host="0.0.0.0", port=port) 