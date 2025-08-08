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
import mediapipe as mp
import cv2
import asyncio
from threading import Thread

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
ENABLE_CHATGPT = os.environ.get('ENABLE_CHATGPT', 'true').lower() == 'true'
REQUIRE_AUTH = os.environ.get('REQUIRE_AUTH', 'false').lower() == 'true'

# Debug logging for environment variables - FORCE REDEPLOY 2024-12-19
print(f"🔍 DEBUG: RAILWAY_ENVIRONMENT = {os.environ.get('RAILWAY_ENVIRONMENT', 'NOT_SET')}")
print(f"🔍 DEBUG: ENABLE_DEEPFACE env var = {os.environ.get('ENABLE_DEEPFACE', 'NOT_SET')}")
print(f"🔍 DEBUG: ENABLE_DEEPFACE parsed = {ENABLE_DEEPFACE}")
print(f"🔍 DEBUG: ENABLE_CHATGPT env var = {os.environ.get('ENABLE_CHATGPT', 'NOT_SET')}")
print(f"🔍 DEBUG: ENABLE_CHATGPT parsed = {ENABLE_CHATGPT}")
print(f"🔍 DEBUG: LOAD_HARVARD_MODEL env var = {os.environ.get('LOAD_HARVARD_MODEL', 'NOT_SET')}")
print(f"🔍 DEBUG: LOAD_HARVARD_MODEL parsed = {LOAD_HARVARD}")
print(f"🔍 DEBUG: REQUIRE_AUTH env var = {os.environ.get('REQUIRE_AUTH', 'NOT_SET')}")
print(f"🔍 DEBUG: REQUIRE_AUTH parsed = {REQUIRE_AUTH}")

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
# if IS_RAILWAY:
    #time.sleep(5)  # Give Railway time to initialize

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

def background_load_models():
    """Load models in background thread - non-blocking"""
    global detector, harvard_model, deepface_ready, models_loading
    
    if models_loading or (detector is not None and (harvard_model is not None or not LOAD_HARVARD) and (deepface_ready or not ENABLE_DEEPFACE)):
        return
    
    models_loading = True
    logger.info("🔄 Background loading models...")
    
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
        logger.info("✅ Background model loading complete!")
        
    except Exception as e:
        logger.error(f"❌ Background model loading failed: {e}")
        models_loading = False

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

# Facial landmark regions using MediaPipe's official predefined connection sets
def build_facial_regions_from_connections():
    """Build facial regions from MediaPipe's official connection sets"""
    import mediapipe as mp
    
    # Get the official MediaPipe Face Mesh connections
    mp_face_mesh = mp.solutions.face_mesh
    FACEMESH_LIPS = mp_face_mesh.FACEMESH_LIPS
    FACEMESH_LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
    FACEMESH_RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
    FACEMESH_LEFT_EYEBROW = mp_face_mesh.FACEMESH_LEFT_EYEBROW
    FACEMESH_RIGHT_EYEBROW = mp_face_mesh.FACEMESH_RIGHT_EYEBROW
    FACEMESH_FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL
    FACEMESH_NOSE = mp_face_mesh.FACEMESH_NOSE
    
    # Hair segmentation using MediaPipe Image Segmenter
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    def get_hair_segmentation():
        """Get hair segmentation using MediaPipe Image Segmenter"""
        base_options = python.BaseOptions(model_asset_path='https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        return vision.ImageSegmenter.create_from_options(options)
    
    def connections_to_ordered_loop(connections):
        """Convert connection pairs to ordered loop of indices"""
        if not connections:
            return []
        
        # Convert frozenset to list of tuples
        connection_list = list(connections)
        
        # Build adjacency list
        adj = {}
        for start, end in connection_list:
            if start not in adj:
                adj[start] = []
            if end not in adj:
                adj[end] = []
            adj[start].append(end)
            adj[end].append(start)
        
        # Find ordered loop
        visited = set()
        loop = []
        
        def dfs(node, path):
            if len(path) > 1 and node == path[0]:
                return path
            if node in visited:
                return None
            visited.add(node)
            path.append(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in path or (len(path) > 2 and neighbor == path[0]):
                    result = dfs(neighbor, path)
                    if result:
                        return result
            return None
        
        # Start from first connection
        if connection_list:
            start_node = connection_list[0][0]
            loop = dfs(start_node, [])
        
        return loop if loop else [conn[0] for conn in connection_list]
    
    return {
    "eyes": {
            "left_eye": connections_to_ordered_loop(FACEMESH_LEFT_EYE),
            "right_eye": connections_to_ordered_loop(FACEMESH_RIGHT_EYE),
            "left_eyebrow": connections_to_ordered_loop(FACEMESH_LEFT_EYEBROW),
            "right_eyebrow": connections_to_ordered_loop(FACEMESH_RIGHT_EYEBROW),
            "color": (0, 0, 255),  # Red (BGR format) - Highly visible
            "style": "dots"  # Unified dots style
    },
    "nose": {
            "nose": connections_to_ordered_loop(FACEMESH_NOSE),
            "color": (0, 255, 255),  # Yellow (BGR format) - Highly visible
            "style": "dots"  # Unified dots style
    },
    "mouth": {
            "lips": connections_to_ordered_loop(FACEMESH_LIPS),
            "color": (0, 255, 0),  # Green (BGR format) - Highly visible
            "style": "dots"  # Unified dots style
    },
    "face_contour": {
            "face_oval": connections_to_ordered_loop(FACEMESH_FACE_OVAL),
            "color": (255, 0, 0),  # Blue (BGR format) - Highly visible
            "style": "dots"  # Unified dots style
        },
        "hair": {
            "hair_mask": [],  # Will be populated by hair segmentation
            "color": (128, 0, 128),  # Purple (BGR format) - Highly visible
            "style": "dots"  # Unified dots style
        }
    }

# Generate facial regions from MediaPipe connections
FACIAL_REGIONS = build_facial_regions_from_connections()

def draw_facemesh(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    overlay = img.copy()
    mp_face_mesh = mp.solutions.face_mesh
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw only dots, with blue/green gradient and opacity
                    n_points = len(face_landmarks.landmark)
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        # Gradient from blue (#4f8cff) to green (#4fd1c5)
                        t = idx / n_points
                        r = int((0x4f * (1-t)) + (0x4f * t))
                        g = int((0x8c * (1-t)) + (0xd1 * t))
                        b = int((0xff * (1-t)) + (0xc5 * t))
                        color = (b, g, r, 180)  # OpenCV uses BGR, alpha=180/255
                        # Draw semi-transparent dot
                        cv2.circle(overlay, (x, y), 2, (b, g, r), -1, lineType=cv2.LINE_AA)
                    # Blend overlay with original for opacity
                img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        _, buffer = cv2.imencode('.jpg', img)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded
    except Exception as e:
        logger.error(f"FaceMesh error: {e}")
        blank = np.zeros_like(img)
        _, buffer = cv2.imencode('.jpg', blank)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded

def detect_hair_segmentation(image_bytes):
    """Detect hair using MediaPipe Image Segmenter"""
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe import Image
        import urllib.request
        import os
        
        logger.info("Starting hair segmentation...")
        
        # Download the hair segmentation model if not already present
        model_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite'
        model_path = 'hair_segmenter.tflite'
        
        if not os.path.exists(model_path):
            logger.info(f"Downloading hair segmentation model from {model_url}")
            urllib.request.urlretrieve(model_url, model_path)
            logger.info("Hair segmentation model downloaded successfully")
        else:
            logger.info("Using cached hair segmentation model")
        
        # Create image segmenter for hair detection
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        segmenter = vision.ImageSegmenter.create_from_options(options)
        
        logger.info("Hair segmenter created successfully")
        
        # Convert image bytes to MediaPipe Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        logger.info(f"Image converted: shape={img.shape}")
        
        # Segment hair
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask
        
        logger.info(f"Segmentation result: mask shape={category_mask.numpy_view().shape}")
        
        # Get hair mask (category 1 is hair)
        hair_mask = (category_mask.numpy_view() == 1).astype(np.uint8) * 255
        
        # Debug: log hair mask info
        hair_pixels = np.sum(hair_mask > 0)
        total_pixels = hair_mask.shape[0] * hair_mask.shape[1]
        hair_percentage = (hair_pixels / total_pixels) * 100
        logger.info(f"Hair segmentation: {hair_pixels} hair pixels ({hair_percentage:.1f}% of image)")
        
        # Validate hair detection - if too much or too little hair is detected, it's likely wrong
        # Typical hair percentages: 5-25% for people with hair, <2% for bald people
        if hair_percentage > 30 or hair_percentage < 2:
            logger.warning(f"Hair segmentation likely incorrect: {hair_percentage:.1f}% detected (expected 2-30%)")
            return None
        
        return hair_mask
    except Exception as e:
        logger.error(f"Hair segmentation error: {e}")
        import traceback
        logger.error(f"Hair segmentation traceback: {traceback.format_exc()}")
        return None

def find_lowest_y_for_each_x(points, x_range, tolerance=2, smoothing_level="medium"):
    """
    Find the lowest Y coordinate for each X coordinate with configurable smoothing.
    This creates a clean bottom edge by ensuring only one Y value per X.
    
    Args:
        points: Array of [x, y] coordinates
        x_range: Array of X coordinates to sample
        tolerance: How close X coordinates can be to be considered "at the same X"
        smoothing_level: "low", "medium", "high", or "ultra" for different smoothing intensities
    
    Returns:
        Array of [x, y] coordinates with exactly one Y per X, smoothed according to level
    """
    # Convert points to numpy array if it isn't already
    points = np.array(points, dtype=np.int32)
    
    # Create a dictionary to store the lowest Y for each X
    x_to_lowest_y = {}
    
    # For each X in our range, find the lowest Y
    for x in x_range:
        x_int = int(x)
        
        # Find all points at this X coordinate (within tolerance)
        points_at_x = []
        for point in points:
            if abs(point[0] - x_int) <= tolerance:
                points_at_x.append(point)
        
        if points_at_x:
            # Use the lowest Y (highest pixel value) among points at this X
            lowest_y = max(point[1] for point in points_at_x)
            x_to_lowest_y[x_int] = lowest_y
    
    # Convert back to array format, sorted by X
    result = []
    for x in sorted(x_to_lowest_y.keys()):
        result.append([x, x_to_lowest_y[x]])
    
    result = np.array(result, dtype=np.int32)
    logger.info(f"find_lowest_y_for_each_x: input {len(points)} points, output {len(result)} points")
    
    # Apply smoothing based on level
    if len(result) > 3:
        try:
            from scipy.ndimage import gaussian_filter1d
            
            # Define smoothing parameters based on level
            smoothing_params = {
                "low": {"sigma": 0.5, "spline_s": 50},
                "medium": {"sigma": 1.0, "spline_s": 100},
                "high": {"sigma": 1.5, "spline_s": 200},
                "ultra": {"sigma": 2.0, "spline_s": 300}
            }
            
            params = smoothing_params.get(smoothing_level, smoothing_params["medium"])
            
            # Apply Gaussian smoothing to Y coordinates
            smoothed_y = gaussian_filter1d(result[:, 1].astype(float), sigma=params["sigma"])
            result = np.column_stack([result[:, 0], smoothed_y.astype(int)])
            
        except ImportError:
            # If scipy is not available, skip smoothing
            pass
    
    return result

def draw_facial_regions(image_bytes):
    """Draw facial regions with clean, minimal, medical-tech style + face mesh dots"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    overlay = img.copy()
    mp_face_mesh = mp.solutions.face_mesh
    
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # FIRST: Draw all face mesh dots (like in draw_facemesh)
                    n_points = len(face_landmarks.landmark)
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        # Gradient from blue (#4f8cff) to green (#4fd1c5)
                        t = idx / n_points
                        r = int((0x4f * (1-t)) + (0x4f * t))
                        g = int((0x8c * (1-t)) + (0xd1 * t))
                        b = int((0xff * (1-t)) + (0xc5 * t))
                        color = (b, g, r, 180)  # OpenCV uses BGR, alpha=180/255
                        # Draw semi-transparent dot
                        cv2.circle(overlay, (x, y), 2, (b, g, r), -1, lineType=cv2.LINE_AA)
                    
                    # SECOND: Draw tessellation edges (subtle mesh lines)
                    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
                    h, w = img.shape[:2]
                    
                    # Draw tessellation lines (brighter and slightly thicker)
                    for a, b in FACEMESH_TESSELATION:
                        if a < len(face_landmarks.landmark) and b < len(face_landmarks.landmark):
                            pt1 = (int(face_landmarks.landmark[a].x * w), int(face_landmarks.landmark[a].y * h))
                            pt2 = (int(face_landmarks.landmark[b].x * w), int(face_landmarks.landmark[b].y * h))
                            cv2.line(overlay, pt1, pt2, (220, 220, 220), 2, cv2.LINE_AA)
                    
                    # THIRD: Draw facial region outlines on top of the dots and tessellation
                    
                    # Eyes: bright modern light blue outline
                    eye_color = (255, 255, 0)  # Bright cyan in BGR
                    for eye_region in ["left_eye", "right_eye"]:
                        if eye_region in FACIAL_REGIONS["eyes"]:
                            landmark_indices = FACIAL_REGIONS["eyes"][eye_region]
                            points = []
                            
                            for idx in landmark_indices:
                                if idx < len(face_landmarks.landmark):
                                    landmark = face_landmarks.landmark[idx]
                                    x = int(landmark.x * img.shape[1])
                                    y = int(landmark.y * img.shape[0])
                                    points.append([x, y])
                            
                            if len(points) > 2:
                                points = np.array(points, dtype=np.int32)
                                # Draw smooth outline with proper curve smoothing
                                if len(points) > 4:
                                    # Use spline interpolation for smooth curves
                                    from scipy.interpolate import splprep, splev
                                    try:
                                        # Close the curve by adding first point at end
                                        closed_points = np.vstack([points, points[0]])
                                        # Fit spline with more smoothing (higher s value)
                                        tck, u = splprep([closed_points[:, 0], closed_points[:, 1]], s=100, per=True)
                                        # Generate smooth curve points with more interpolation
                                        new_points = np.array(splev(np.linspace(0, 1, 200), tck)).T.astype(np.int32)
                                        cv2.polylines(overlay, [new_points], True, eye_color, 3, cv2.LINE_AA)
                                    except:
                                        # Fallback to simple polyline if spline fails
                                        cv2.polylines(overlay, [points], True, eye_color, 3, cv2.LINE_AA)
                                else:
                                    cv2.polylines(overlay, [points], True, eye_color, 3, cv2.LINE_AA)
                    
                    # Eyebrows: use lowest point logic for clean bottom edge
                    eyebrow_color = (255, 255, 0)  # Bright cyan in BGR
                    for eyebrow_region in ["left_eyebrow", "right_eyebrow"]:
                        if eyebrow_region in FACIAL_REGIONS["eyes"]:
                            landmark_indices = FACIAL_REGIONS["eyes"][eyebrow_region]
                            points = []
                            
                            for idx in landmark_indices:
                                if idx < len(face_landmarks.landmark):
                                    landmark = face_landmarks.landmark[idx]
                                    x = int(landmark.x * img.shape[1])
                                    y = int(landmark.y * img.shape[0])
                                    points.append([x, y])
                            
                            if len(points) > 2:
                                points = np.array(points, dtype=np.int32)
                                
                                # Use the simple function to find lowest Y for each X with high smoothing for eyebrows
                                min_x = np.min(points[:, 0])
                                max_x = np.max(points[:, 0])
                                x_range = np.linspace(min_x, max_x, 60)
                                logger.info(f"Eyebrow: processing {len(points)} points, X range: {min_x} to {max_x}")
                                eyebrow_bottom_points = find_lowest_y_for_each_x(points, x_range, tolerance=2, smoothing_level="ultra")
                                logger.info(f"Eyebrow: function returned {len(eyebrow_bottom_points)} points")
                                
                                # Draw the clean eyebrow bottom edge
                                if len(eyebrow_bottom_points) > 3:
                                    # Apply spline smoothing for ultra-smooth curves
                                    try:
                                        from scipy.interpolate import splprep, splev
                                        # Fit spline with high smoothing parameter
                                        tck, u = splprep([eyebrow_bottom_points[:, 0], eyebrow_bottom_points[:, 1]], s=150)
                                        # Generate very smooth curve with more interpolation points
                                        new_points = np.array(splev(np.linspace(0, 1, 300), tck)).T.astype(np.int32)
                                        cv2.polylines(overlay, [new_points], False, eyebrow_color, 3, cv2.LINE_AA)
                                        logger.info(f"Eyebrow bottom edge drawn with {len(eyebrow_bottom_points)} points, smoothed to 300 points")
                                    except:
                                        # Fallback to simple polyline if spline fails
                                        cv2.polylines(overlay, [eyebrow_bottom_points], False, eyebrow_color, 3, cv2.LINE_AA)
                                        logger.info(f"Eyebrow bottom edge drawn with {len(eyebrow_bottom_points)} points (no smoothing)")
                                else:
                                    logger.warning(f"Not enough eyebrow bottom points: {len(eyebrow_bottom_points)}")
                    
                    # Nose: base and bridge using lowest point logic
                    if "nose" in FACIAL_REGIONS["nose"]:
                        nose_color = (255, 255, 0)  # Bright cyan in BGR
                        landmark_indices = FACIAL_REGIONS["nose"]["nose"]
                        points = []
                        
                        for idx in landmark_indices:
                            if idx < len(face_landmarks.landmark):
                                landmark = face_landmarks.landmark[idx]
                                x = int(landmark.x * img.shape[1])
                                y = int(landmark.y * img.shape[0])
                                points.append([x, y])
                        
                        if len(points) > 2:
                            points = np.array(points, dtype=np.int32)
                            
                            # Draw nose base (bottom edge)
                            min_x = np.min(points[:, 0])
                            max_x = np.max(points[:, 0])
                            x_range = np.linspace(min_x, max_x, 40)
                            logger.info(f"Nose: processing {len(points)} points, X range: {min_x} to {max_x}")
                            nose_base_points = find_lowest_y_for_each_x(points, x_range, tolerance=2, smoothing_level="high")
                            logger.info(f"Nose base: found {len(nose_base_points)} points")
                            
                            if len(nose_base_points) > 3:
                                # Draw the nose base as a single continuous line
                                # Sort points by X coordinate to ensure proper ordering
                                nose_base_points = nose_base_points[nose_base_points[:, 0].argsort()]
                                
                                # Apply spline smoothing for the nose base
                                try:
                                    from scipy.interpolate import splprep, splev
                                    tck, u = splprep([nose_base_points[:, 0], nose_base_points[:, 1]], s=50)
                                    new_points = np.array(splev(np.linspace(0, 1, 200), tck)).T.astype(np.int32)
                                    cv2.polylines(overlay, [new_points], False, nose_color, 3, cv2.LINE_AA)
                                    logger.info(f"Nose base drawn with {len(nose_base_points)} points, smoothed to 200 points")
                                except:
                                    cv2.polylines(overlay, [nose_base_points], False, nose_color, 3, cv2.LINE_AA)
                                    logger.info(f"Nose base drawn with {len(nose_base_points)} points (no smoothing)")
                    
                    # Lips: magenta outline with slight interior shading
                    if "lips" in FACIAL_REGIONS["mouth"]:
                        lip_color = (255, 255, 0)  # Bright cyan in BGR
                        landmark_indices = FACIAL_REGIONS["mouth"]["lips"]
                        points = []
                        
                        for idx in landmark_indices:
                            if idx < len(face_landmarks.landmark):
                                landmark = face_landmarks.landmark[idx]
                                x = int(landmark.x * img.shape[1])
                                y = int(landmark.y * img.shape[0])
                                points.append([x, y])
                        
                        if len(points) > 2:
                            points = np.array(points, dtype=np.int32)
                            # Draw smooth lip outline with proper curve smoothing
                            if len(points) > 4:
                                # Use spline interpolation for smooth curves
                                from scipy.interpolate import splprep, splev
                                try:
                                    # Close the curve by adding first point at end
                                    closed_points = np.vstack([points, points[0]])
                                    # Fit spline with more smoothing (higher s value)
                                    tck, u = splprep([closed_points[:, 0], closed_points[:, 1]], s=100, per=True)
                                    # Generate smooth curve points with more interpolation
                                    new_points = np.array(splev(np.linspace(0, 1, 200), tck)).T.astype(np.int32)
                                    cv2.polylines(overlay, [new_points], True, lip_color, 3, cv2.LINE_AA)
                                except:
                                    # Fallback to simple polyline if spline fails
                                    cv2.polylines(overlay, [points], True, lip_color, 3, cv2.LINE_AA)
                            else:
                                cv2.polylines(overlay, [points], True, lip_color, 3, cv2.LINE_AA)
                    
                    # Face contour: light blue outline
                    if "face_oval" in FACIAL_REGIONS["face_contour"]:
                        contour_color = (255, 255, 0)  # Bright cyan in BGR
                        landmark_indices = FACIAL_REGIONS["face_contour"]["face_oval"]
                        points = []
                        
                        for idx in landmark_indices:
                            if idx < len(face_landmarks.landmark):
                                landmark = face_landmarks.landmark[idx]
                                x = int(landmark.x * img.shape[1])
                                y = int(landmark.y * img.shape[0])
                                points.append([x, y])
                        
                        if len(points) > 2:
                            points = np.array(points, dtype=np.int32)
                            # Draw smooth face contour with proper curve smoothing
                            if len(points) > 4:
                                # Use spline interpolation for smooth curves
                                from scipy.interpolate import splprep, splev
                                try:
                                    # Close the curve by adding first point at end
                                    closed_points = np.vstack([points, points[0]])
                                    # Fit spline with more smoothing (higher s value)
                                    tck, u = splprep([closed_points[:, 0], closed_points[:, 1]], s=100, per=True)
                                    # Generate smooth curve points with more interpolation
                                    new_points = np.array(splev(np.linspace(0, 1, 200), tck)).T.astype(np.int32)
                                    cv2.polylines(overlay, [new_points], True, contour_color, 3, cv2.LINE_AA)
                                except:
                                    # Fallback to simple polyline if spline fails
                                    cv2.polylines(overlay, [points], True, contour_color, 3, cv2.LINE_AA)
                            else:
                                cv2.polylines(overlay, [points], True, contour_color, 3, cv2.LINE_AA)
                    
                    # Hairline: from hair segmentation mask, trace the actual hairline boundary
                    hair_mask = detect_hair_segmentation(image_bytes)
                    if hair_mask is not None:
                        logger.info(f"Drawing hairline: mask shape={hair_mask.shape}")
                        hair_mask_resized = cv2.resize(hair_mask, (img.shape[1], img.shape[0]))
                        
                        # Find the hairline by taking the lowest Y coordinate for each X
                        hairline_points = []
                        img_height, img_width = hair_mask_resized.shape
                        
                        # Only process the upper portion of the image (forehead area)
                        max_y = int(img_height * 0.4)  # Upper 40% of image
                        
                        for x in range(img_width):
                            # Find the highest Y coordinate (lowest pixel value) for this X - this is the hairline
                            highest_y = None
                            for y in range(max_y):
                                if hair_mask_resized[y, x] > 0:  # If hair pixel found
                                    highest_y = y  # Keep updating to find the lowest hair pixel (highest Y)
                            
                            if highest_y is not None:
                                hairline_points.append([x, highest_y])
                        
                        # Use the simple function to find lowest Y for each X (for hairline, we want the highest Y)
                        if len(hairline_points) > 15:
                            hairline_points = np.array(hairline_points, dtype=np.int32)
                            # For hairline, we want the highest Y (lowest on screen), so we invert the logic
                            # The hairline points are already the "lowest" hair pixels, so we use them directly
                            x_range = np.linspace(0, img_width-1, 100)  # Sample across full width
                            smoothed_hairline = find_lowest_y_for_each_x(hairline_points, x_range, tolerance=3, smoothing_level="medium")
                        
                        # Use the smoothed hairline from our function
                        if len(smoothed_hairline) > 5:
                            # Apply final spline smoothing for ultra-smooth curves
                            hairline_color = (255, 255, 0)  # Bright cyan in BGR
                            try:
                                from scipy.interpolate import splprep, splev
                                # Fit spline with high smoothing parameter
                                tck, u = splprep([smoothed_hairline[:, 0], smoothed_hairline[:, 1]], s=200)
                                # Generate very smooth curve with more interpolation points
                                new_points = np.array(splev(np.linspace(0, 1, 400), tck)).T.astype(np.int32)
                                cv2.polylines(overlay, [new_points], False, hairline_color, 3, cv2.LINE_AA)
                                logger.info(f"Ultra-smooth hairline drawn with {len(smoothed_hairline)} points, smoothed to 400 points")
                            except:
                                # Fallback to simple polyline if spline fails
                                cv2.polylines(overlay, [smoothed_hairline], False, hairline_color, 3, cv2.LINE_AA)
                                logger.info(f"Hairline drawn with {len(smoothed_hairline)} points (no spline smoothing)")
                        else:
                            logger.warning(f"Not enough smoothed hairline points: {len(smoothed_hairline)}")
                    else:
                        logger.warning("Hair mask is None - hair segmentation failed")
                
                # Blend overlay with original for more visible tessellation effect
                img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        
        _, buffer = cv2.imencode('.jpg', img)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded
    except Exception as e:
        logger.error(f"Facial regions error: {e}")
        blank = np.zeros_like(img)
        _, buffer = cv2.imencode('.jpg', blank)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded

# FastAPI app without startup model loading
app = FastAPI(
    title="TrueAge API",
    description="TrueAge uses advanced AI models to estimate your biological age and perceived age from a single facial photo. Upload your selfie to discover insights about your health and aging, powered by state-of-the-art deep learning and facial analysis technology.",
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

# Add compression middleware for better performance
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/")
async def root():
    """Root endpoint serving the React Native web build - serves immediately"""
    web_build_path = "FaceAgeApp/web-build"
    if os.path.exists(f"{web_build_path}/index.html"):
        return FileResponse(f"{web_build_path}/index.html")
    return {"message": "TrueAge API", "status": "running", "server_ready": True}

@app.get("/health")
async def health_check():
    """Health check endpoint - responds immediately"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "environment": "railway" if IS_RAILWAY else "local",
        "server_ready": True,
        "models_loading": models_loading
    }

@app.get("/api/health")
async def api_health_check():
    # Don't trigger model loading, just report current status
    harvard_status = harvard_model is not None
    deepface_status = deepface_ready if 'deepface_ready' in globals() else False
    chatgpt_status = bool(os.environ.get('OPENAI_API_KEY')) and ENABLE_CHATGPT
    return {
        "status": "healthy",
        "models": {
            "harvard": harvard_status,
            "deepface": deepface_status,
            "chatgpt": chatgpt_status
        },
        "models_loading": models_loading,
        "ready_for_analysis": harvard_status or deepface_status,
        "require_auth": REQUIRE_AUTH,
        "settings": {
            "enable_deepface": ENABLE_DEEPFACE,
            "enable_chatgpt": ENABLE_CHATGPT,
            "load_harvard": LOAD_HARVARD
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
        high_confidence_faces = [f for f in faces if f['confidence'] >= 0.95]
        
        if not high_confidence_faces:
            return AnalyzeResponse(
                success=False,
                faces=[],
                message="No high-confidence faces detected (≥95% confidence required)"
            )
        
        results = []
        
        for i, face in enumerate(high_confidence_faces):
            logger.info(f"--- Processing face {i} ---")
            try:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                logger.info(f"Face {i}: extracting crop...")
                # Original crop for Harvard/DeepFace (existing logic)
                face_crop = img_array[y:y+h, x:x+w]
                
                # Enhanced crop for ChatGPT with safe clipping + square padding
                H, W = img_array.shape[:2]
                x = max(0, x); y = max(0, y)
                x2 = min(W, x + w); y2 = min(H, y + h)
                cx, cy = (x + x2)//2, (y + y2)//2
                side = int(1.3 * max(x2 - x, y2 - y))  # 30% padding for hair/context
                x1 = max(0, cx - side//2); y1 = max(0, cy - side//2)
                x2 = min(W, x1 + side); y2 = min(H, y1 + side)
                chatgpt_face_crop = img_array[y1:y2, x1:x2]
                
                logger.info(f"Face {i}: original crop shape={face_crop.shape}, ChatGPT crop shape={chatgpt_face_crop.shape}")
                try:
                    logger.info(f"Face {i}: converting crop to base64...")
                    # Use the padded ChatGPT crop for display (includes hair/context)
                    face_pil = Image.fromarray(chatgpt_face_crop)
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
                if ENABLE_CHATGPT:
                    try:
                        logger.info(f"Face {i}: Calling estimate_age_chatgpt...")
                        # Create base64 from ChatGPT-specific crop (with padding for hair/context)
                        chatgpt_face_pil = Image.fromarray(chatgpt_face_crop)
                        chatgpt_face_buffer = BytesIO()
                        chatgpt_face_pil.save(chatgpt_face_buffer, format='PNG')
                        chatgpt_face_base64 = base64.b64encode(chatgpt_face_buffer.getvalue()).decode('utf-8')
                        
                        chatgpt_response = estimate_age_chatgpt(chatgpt_face_base64)
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
                else:
                    logger.info(f"Face {i}: 🚫 ChatGPT prediction skipped (disabled)")
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

@app.post("/facemesh-overlay")
async def facemesh_overlay(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        # Use the combined visualization (dots + facial regions) as the default
        result = draw_facial_regions(image_bytes)
        return {"image_base64": result}
    except Exception as e:
        logger.error(f"/facemesh-overlay error: {e}")
        # Return a blank image or error placeholder
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return {"image_base64": encoded}



# Mount static files for React Native web build with caching
class CachedStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> FileResponse:
        response = super().file_response(*args, **kwargs)
        # Add caching headers for better performance
        if any(args[0].endswith(ext) for ext in ['.js', '.css', '.png', '.jpg', '.ico']):
            response.headers["Cache-Control"] = "public, max-age=31536000"  # 1 year
        else:
            response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour
        return response

# Check if we're in development mode
IS_DEVELOPMENT = os.environ.get("DEVELOPMENT", "false").lower() == "true"

web_build_path = "FaceAgeApp/web-build"
if os.path.exists(web_build_path) and not IS_DEVELOPMENT:
    # Production: Serve static build
    app.mount("/", CachedStaticFiles(directory=web_build_path, html=True), name="static")
    print("🏭 Production mode: Serving static build")
else:
    # Development: Serve API only, frontend runs on Expo dev server
    @app.get("/")
    async def dev_root():
        return {
            "message": "Bio Age API Server",
            "status": "running",
            "development": True,
            "frontend_url": "http://localhost:19006",
            "api_docs": "/docs",
            "health": "/api/health"
        }
    print("🔧 Development mode: API server only (frontend on Expo dev server)")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting Bio Age Estimator on port {port}")
    print(f"📁 Web build path: {web_build_path}")
    print(f"🌐 Frontend available: {os.path.exists(web_build_path)}")
    print(f"🔧 Railway environment: {IS_RAILWAY}")
    print(f"📊 Harvard model enabled: {LOAD_HARVARD}")
    print(f"🤖 DeepFace enabled: {ENABLE_DEEPFACE}")
    print(f"🤖 ChatGPT enabled: {ENABLE_CHATGPT}")
    print(f"⚡ Using background model loading for instant startup")
    print('OpenAI version at runtime:', openai.__version__)
    
    # Start background model loading immediately (non-blocking)
    print("🔄 Starting background model loading...")
    model_thread = Thread(target=background_load_models, daemon=True)
    model_thread.start()
    
    print("✅ Server starting immediately - models loading in background")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300,  # 5 minutes keep-alive
        timeout_graceful_shutdown=300  # 5 minutes graceful shutdown
    ) 