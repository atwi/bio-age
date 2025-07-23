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
import traceback
import psutil

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

# Debug logging for environment variables - FORCE REDEPLOY 2024-12-19
print(f"🔍 DEBUG: RAILWAY_ENVIRONMENT = {os.environ.get('RAILWAY_ENVIRONMENT', 'NOT_SET')}")
print(f"🔍 DEBUG: ENABLE_DEEPFACE env var = {os.environ.get('ENABLE_DEEPFACE', 'NOT_SET')}")
print(f"🔍 DEBUG: ENABLE_DEEPFACE parsed = {ENABLE_DEEPFACE}")
print(f"🔍 DEBUG: LOAD_HARVARD_MODEL env var = {os.environ.get('LOAD_HARVARD_MODEL', 'NOT_SET')}")
print(f"🔍 DEBUG: LOAD_HARVARD_MODEL parsed = {LOAD_HARVARD}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if IS_RAILWAY:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Add startup delay for Railway
if IS_RAILWAY:
    time.sleep(5)  # Give Railway time to initialize

def log_memory_usage(context=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    logger.info(f"[MEMORY] {context} Memory usage: {mem:.2f} MB")

class FaceResult(BaseModel):
    face_id: int
    age_harvard: Optional[float] = None
    age_deepface: Optional[float] = None
    age_chatgpt: Optional[int] = None
    chatgpt_factors: Optional[dict] = None  # New: factor breakdown
    confidence: float
    face_crop_base64: str
    chatgpt_raw_response: Optional[str] = None  # Raw OpenAI response
    chatgpt_fallback_text: Optional[str] = None  # Fallback text if used
    chatgpt_error: Optional[str] = None  # Error if any

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
            else:
                logger.warning("⚠️ Harvard model failed to load, continuing without it")
            gc.collect()
        
        # Test DeepFace last (largest) - only if enabled
        if ENABLE_DEEPFACE and not deepface_ready:
            logger.info("Initializing DeepFace...")
            deepface_ready = test_deepface()
            if deepface_ready:
                logger.info("✅ DeepFace initialized")
            else:
                logger.warning("⚠️ DeepFace failed to initialize, continuing without it")
            gc.collect()
        elif not ENABLE_DEEPFACE:
            logger.info("🚫 DeepFace disabled by configuration")
            deepface_ready = True  # Mark as ready to skip loading
        
        models_loading = False
        
        # Check what we have available
        has_face_detector = detector is not None
        has_harvard = harvard_model is not None
        has_deepface = deepface_ready or not ENABLE_DEEPFACE
        
        logger.info(f"📊 Model status: Face detector: {has_face_detector}, Harvard: {has_harvard}, DeepFace: {has_deepface}")
        
        # We need face detector and at least one age estimation model
        if has_face_detector and (has_harvard or has_deepface):
            logger.info("✅ Sufficient models loaded for face analysis")
            return True
        else:
            logger.warning("⚠️ Limited models available - face analysis may be restricted")
            return has_face_detector  # Return True if we at least have face detection
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        models_loading = False
        return detector is not None  # Return True if we at least have face detection



def load_harvard_model():
    """Load Harvard model - should be downloaded during build phase"""
    import tensorflow as tf
    
    # Model should already be downloaded during build phase
    possible_paths = [
        "model_saved_tf",
        "./model_saved_tf",
        "FaceAge/model_saved_tf",
        "./FaceAge/model_saved_tf"
    ]
    
    # Debug: list all files in current directory
    logger.info(f"Current directory contents: {os.listdir('.')}")
    
    # Check if we're in a subdirectory
    if os.path.exists("FaceAge"):
        logger.info(f"FaceAge directory contents: {os.listdir('FaceAge')}")
    
    for model_path in possible_paths:
        logger.info(f"Checking path: {model_path} (exists: {os.path.exists(model_path)})")
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading Harvard model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                logger.info("✅ Harvard model loaded successfully")
                log_memory_usage("After Harvard model load")
                return model
            except Exception as e:
                logger.warning(f"Failed to load from {model_path}: {e}")
                continue
    
    # If model not found, try to download it at runtime (fallback)
    logger.warning("❌ Harvard model not found! Attempting runtime download...")
    try:
        import gdown
        import zipfile
        
        logger.info("📥 Downloading Harvard model at runtime...")
        MODEL_ZIP = 'model_saved_tf.zip'
        MODEL_DIR = 'model_saved_tf'
        
        # Download the model
        gdown.download(
            'https://drive.google.com/uc?id=12wNpYBz3j5mP9mt6S_ZH4k0sI6dVDeVV', 
            MODEL_ZIP, 
            quiet=False
        )
        
        if os.path.exists(MODEL_ZIP):
            logger.info("📦 Extracting Harvard model...")
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            # Clean up zip file
            os.remove(MODEL_ZIP)
            
            # Try to load the model
            if os.path.exists(MODEL_DIR):
                logger.info(f"Loading Harvard model from runtime download: {MODEL_DIR}")
                model = tf.keras.models.load_model(MODEL_DIR)
                logger.info("✅ Harvard model loaded successfully from runtime download")
                log_memory_usage("After Harvard model load (runtime download)")
                return model
            else:
                logger.error("❌ Model directory not found after runtime download")
        else:
            logger.error("❌ Model zip file not downloaded")
            
    except Exception as e:
        logger.error(f"❌ Runtime model download failed: {e}")
    
    logger.error("❌ Harvard model not found! Build phase download must have failed.")
    logger.error("Expected model_saved_tf directory to exist from build phase.")
    return None

def test_deepface():
    """Test DeepFace initialization"""
    log_memory_usage("Before DeepFace load")
    try:
        from deepface import DeepFace
        log_memory_usage("After DeepFace import")
        # Create minimal test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test DeepFace
        result = DeepFace.analyze(
            test_img,
            actions=['age'],
            enforce_detection=False,
            silent=True
        )
        log_memory_usage("After DeepFace analyze test")
        # Clean up
        del test_img, result
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"DeepFace test failed: {e}")
        return False

def estimate_age_chatgpt(image_base64: str) -> dict:
    """Estimate age and aging factors using ChatGPT Vision with function calling. Returns dict with result, raw, fallback, error."""
    result = {
        'function_args': None,
        'raw_response': None,
        'fallback_text': None,
        'error': None
    }
    try:
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "estimate_age_factors",
                    "description": "Estimate the age and key aging factors for the person in the image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skin_texture": {
                                "type": "object",
                                "properties": {
                                    "age_rating": {"type": "integer", "description": "Age rating for skin texture (years)"},
                                    "explanation": {"type": "string", "description": "Brief explanation"}
                                },
                                "required": ["age_rating", "explanation"]
                            },
                            "skin_tone": {
                                "type": "object",
                                "properties": {
                                    "age_rating": {"type": "integer", "description": "Age rating for skin tone (years)"},
                                    "explanation": {"type": "string", "description": "Brief explanation"}
                                },
                                "required": ["age_rating", "explanation"]
                            },
                            "hair": {
                                "type": "object",
                                "properties": {
                                    "age_rating": {"type": "integer", "description": "Age rating for hair (years)"},
                                    "explanation": {"type": "string", "description": "Brief explanation"}
                                },
                                "required": ["age_rating", "explanation"]
                            },
                            "facial_volume": {
                                "type": "object",
                                "properties": {
                                    "age_rating": {"type": "integer", "description": "Age rating for facial volume (years)"},
                                    "explanation": {"type": "string", "description": "Brief explanation"}
                                },
                                "required": ["age_rating", "explanation"]
                            },
                            "overall_perceived_age": {"type": "integer", "description": "Overall perceived age in years (18-100)", "minimum": 18, "maximum": 100}
                        },
                        "required": ["skin_texture", "skin_tone", "hair", "facial_volume", "overall_perceived_age"]
                    }
                }
            }
        ]
        image_url = f"data:image/jpeg;base64,{image_base64}"
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)
        system_message = {"role": "system", "content": "You are an API that must always return all required fields for the function call. Never leave any field blank or empty. If unsure, make your best estimate."}
        prompt = (
            "Given the following face image, estimate the 'age rating' (in years) for each of these factors of aging, and provide a brief, consistent explanation for each: "
            "- Skin Texture (wrinkles, pores, smoothness)\n"
            "- Skin Tone (evenness, sun damage, spots)\n"
            "- Hair (hairline, thickness, greying)\n"
            "- Facial Volume (fullness, sagging, fat loss)\n"
            "\n"
            "Your ratings should reflect the genuine perceived age as judged by an attentive, unbiased human observer, based solely on visible facial features, skin, and hair. "
            "Do not attempt to be polite, cautious, or flattering, and do not use medical or technical criteria. "
            "Ignore temporary or artificial factors (such as lighting, makeup, or camera angle) unless they are extreme. "
            "Assume the image is representative of the person's usual appearance.\n"
            "\n"
            "Return your answer in this exact JSON format: { 'skin_texture': { 'age_rating': <number>, 'explanation': '<brief>' }, ... , 'overall_perceived_age': <number> } "
            "All numbers must be in years. Explanations must be brief and neutral."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                system_message,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            tools=functions,
            tool_choice={"type": "function", "function": {"name": "estimate_age_factors"}},
            max_tokens=300
        )
        result['raw_response'] = str(response)
        logger.info(f"ChatGPT Vision raw response: {response}")
        if response.choices[0].message.tool_calls:
            import json
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            result['function_args'] = function_args
            logger.info(f"ChatGPT Vision function_args: {function_args}")
            required = ["skin_texture", "skin_tone", "hair", "facial_volume", "overall_perceived_age"]
            if all(k in function_args for k in required):
                return result
        response_text = response.choices[0].message.content
        result['fallback_text'] = response_text
        logger.info(f"ChatGPT Vision fallback text: {response_text}")
        import json
        try:
            parsed = json.loads(response_text)
            logger.info(f"ChatGPT Vision parsed fallback: {parsed}")
            required = ["skin_texture", "skin_tone", "hair", "facial_volume", "overall_perceived_age"]
            if all(k in parsed for k in required):
                result['function_args'] = parsed
                return result
        except Exception as e:
            logger.warning(f"ChatGPT Vision fallback JSON parse failed: {e}")
            result['error'] = f"Fallback JSON parse failed: {e}"
        return result
    except Exception as e:
        logger.error(f"ChatGPT age estimation failed: {e}")
        result['error'] = str(e)
        return result

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
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "environment": "railway" if IS_RAILWAY else "local",
        "ready": True
    }

@app.get("/api/health")
async def api_health_check():
    # Ensure models are loaded before reporting status
    lazy_load_models()
    harvard_status = harvard_model is not None
    deepface_status = deepface_ready if 'deepface_ready' in globals() else False
    chatgpt_status = bool(os.environ.get('OPENAI_API_KEY'))
    return {
        "status": "healthy",
        "models": {
            "harvard": harvard_status,
            "deepface": deepface_status,
            "chatgpt": chatgpt_status
        }
    }

@app.post("/api/analyze-face", response_model=AnalyzeResponse)
async def analyze_face(file: UploadFile = File(...)):
    log_memory_usage("Before analyze_face")
    try:
        # Lazy load models on first request
        if not lazy_load_models():
            # Check which models are available
            available_models = []
            if detector is not None:
                available_models.append("face_detector")
            if harvard_model is not None:
                available_models.append("harvard_model")
            if deepface_ready or not ENABLE_DEEPFACE:
                available_models.append("deepface")
            
            if not available_models:
                raise HTTPException(status_code=500, detail="No models available for face analysis")
            else:
                logger.warning(f"Limited models available: {available_models}")
                # Continue with available models
        
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
            logger.info(f"--- Processing face {i} ---")
            try:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                logger.info(f"Face {i}: extracting crop...")
                face_crop = img_array[y:y+h, x:x+w]
                logger.info(f"Face {i}: crop shape={face_crop.shape}, dtype={face_crop.dtype}")
                try:
                    logger.info(f"Face {i}: converting crop to base64...")
                    face_pil = Image.fromarray(face_crop)
                    face_buffer = BytesIO()
                    face_pil.save(face_buffer, format='PNG')
                    face_base64 = base64.b64encode(face_buffer.getvalue()).decode('utf-8')
                    logger.info(f"Face {i}: base64 (first 100 chars): {face_base64[:100]}")
                except Exception as e:
                    logger.error(f"Face {i}: Error creating base64: {e}\n{traceback.format_exc()}")
                    face_base64 = ''
                harvard_age = None
                if harvard_model is not None:
                    try:
                        logger.info(f"Face {i}: running Harvard model...")
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
                        logger.info(f"Face {i}: Harvard age={harvard_age}")
                    except Exception as e:
                        logger.warning(f"Face {i}: Harvard prediction failed: {e}\n{traceback.format_exc()}")
                deepface_age = None
                if ENABLE_DEEPFACE and deepface_ready:
                    try:
                        logger.info(f"Face {i}: running DeepFace model...")
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
                        logger.info(f"Face {i}: DeepFace age={deepface_age}")
                    except Exception as e:
                        logger.warning(f"Face {i}: DeepFace prediction failed: {e}\n{traceback.format_exc()}")
                elif not ENABLE_DEEPFACE:
                    logger.info(f"Face {i}: 🚫 DeepFace prediction skipped (disabled)")
                chatgpt_result = None
                chatgpt_raw = None
                chatgpt_fallback = None
                chatgpt_error = None
                try:
                    logger.info(f"Face {i}: Calling estimate_age_chatgpt...")
                    chatgpt_response = estimate_age_chatgpt(face_base64)
                    chatgpt_result = chatgpt_response.get('function_args')
                    chatgpt_raw = chatgpt_response.get('raw_response')
                    chatgpt_fallback = chatgpt_response.get('fallback_text')
                    chatgpt_error = chatgpt_response.get('error')
                    logger.info(f"Face {i}: ChatGPT Vision result: {chatgpt_result}")
                    if chatgpt_error:
                        logger.error(f"Face {i}: ChatGPT error: {chatgpt_error}")
                except Exception as e:
                    logger.error(f"Face {i}: Exception in estimate_age_chatgpt: {e}\n{traceback.format_exc()}")
                    chatgpt_error = str(e)
                chatgpt_age = None
                chatgpt_factors = None
                if chatgpt_result:
                    chatgpt_age = chatgpt_result.get("overall_perceived_age")
                    chatgpt_factors = chatgpt_result
                else:
                    logger.error(f"Face {i}: ChatGPT Vision failed or returned no result.")
                logger.info(f"Face {i}: Appending result...")
                results.append(FaceResult(
                    face_id=i,
                    age_harvard=harvard_age,
                    age_deepface=deepface_age,
                    age_chatgpt=chatgpt_age,
                    chatgpt_factors=chatgpt_factors,
                    confidence=face['confidence'],
                    face_crop_base64=face_base64,
                    chatgpt_raw_response=chatgpt_raw,
                    chatgpt_fallback_text=chatgpt_fallback,
                    chatgpt_error=chatgpt_error
                ))
                logger.info(f"--- Finished face {i} ---\n")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}\n{traceback.format_exc()}")
                continue
        
        log_memory_usage("After analyze_face")
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
    print('OpenAI version at runtime:', openai.__version__)
    
    # Add startup delay for Railway to ensure proper initialization
    if IS_RAILWAY:
        print("⏳ Railway startup delay: 30 seconds")
        time.sleep(30)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300,  # 5 minutes keep-alive
        timeout_graceful_shutdown=300  # 5 minutes graceful shutdown
    ) 