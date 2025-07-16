#!/usr/bin/env python3
"""
Main deployment script for Railway/Render
Serves both the FastAPI backend and React Native web frontend
Railway-optimized with graceful fallback and memory optimization
"""

import os
import sys
import gc
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Setup logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Detect Railway environment
IS_RAILWAY = 'RAILWAY_ENVIRONMENT' in os.environ
IS_CLOUD = IS_RAILWAY or 'DYNO' in os.environ

if IS_CLOUD:
    logger.info("🌐 Running in cloud environment (Railway/Heroku)")
    # Cloud-specific optimizations
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
else:
    logger.info("🖥️ Running locally")

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the API backend and its initialization functions
import api_backend
from api_backend import app as api_app, init_face_detector, load_harvard_model, test_deepface

# Create the main app
app = FastAPI(title="Bio Age Estimator", description="Full-stack deployment")

# Railway-optimized model initialization
def initialize_models():
    """Initialize models with graceful fallback for Railway"""
    logger.info("🔄 Initializing models...")
    
    model_status = {
        'face_detector': False,
        'harvard_model': False,
        'deepface': False
    }
    
    # Initialize face detector (essential)
    try:
        if init_face_detector():
            logger.info("✅ Face detector initialized successfully")
            model_status['face_detector'] = True
        else:
            logger.warning("⚠️ Face detector initialization failed")
    except Exception as e:
        logger.error(f"❌ Face detector failed: {e}")
    
    # Initialize Harvard model (optional for Railway)
    try:
        if not IS_RAILWAY or os.environ.get('LOAD_HARVARD_MODEL', 'false').lower() == 'true':
            if load_harvard_model():
                logger.info("✅ Harvard model loaded successfully")
                model_status['harvard_model'] = True
            else:
                logger.warning("⚠️ Harvard model not available")
        else:
            logger.info("ℹ️ Harvard model disabled for Railway deployment")
    except Exception as e:
        logger.error(f"❌ Harvard model failed: {e}")
    
    # Test DeepFace (essential)
    try:
        if test_deepface():
            logger.info("✅ DeepFace test successful")
            model_status['deepface'] = True
        else:
            logger.warning("⚠️ DeepFace not available")
    except Exception as e:
        logger.error(f"❌ DeepFace failed: {e}")
    
    # Force garbage collection
    gc.collect()
    
    logger.info("🚀 API server ready!")
    
    # Verify the models are set in api_backend
    logger.info(f"📊 Model status: {model_status}")
    logger.info(f"🔍 Face detector: {api_backend.face_detector is not None}")
    logger.info(f"🎯 Harvard model: {api_backend.harvard_model is not None}")
    
    return model_status

# Mount the API backend under /api
app.mount("/api", api_app)

# Check if React Native web build exists
web_build_path = Path("FaceAgeApp/dist")
logger.info(f"📁 Web build path: {web_build_path}")
logger.info(f"🌐 Frontend available: {web_build_path.exists()}")

# Health check for the main app (define after web_build_path)
@app.get("/health")
async def health_check():
    """Health check for the main deployment"""
    return {
        "status": "healthy",
        "service": "bio-age-estimator",
        "frontend": "available" if web_build_path.exists() else "not_built",
        "backend": "available",
        "environment": "railway" if IS_RAILWAY else "local",
        "models": {
            "face_detector": api_backend.face_detector is not None,
            "harvard_model": api_backend.harvard_model is not None
        }
    }

if web_build_path.exists():
    # Serve static files (React Native web build)
    app.mount("/static", StaticFiles(directory=str(web_build_path)), name="static")
    
    # Serve the React Native app for all other routes (except health)
    @app.get("/{path:path}")
    async def serve_react_app(path: str):
        """Serve the React Native web app"""
        # Skip health endpoint
        if path == "health":
            return await health_check()
            
        file_path = web_build_path / path
        
        # If file exists, serve it
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # Otherwise, serve index.html (SPA routing)
        index_path = web_build_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        
        raise HTTPException(status_code=404, detail="Not found")
    
    # Root route serves the React app
    @app.get("/")
    async def serve_root():
        """Serve the React Native web app root"""
        index_path = web_build_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "React Native web build not found. API is available at /api"}

else:
    # Fallback if web build doesn't exist
    @app.get("/")
    async def root():
        return {
            "message": "Bio Age Estimator API", 
            "api_docs": "/docs",
            "api_health": "/api/health",
            "note": "React Native web build not found. Run 'cd FaceAgeApp && npx expo export --platform web' to build the frontend.",
            "environment": "railway" if IS_RAILWAY else "local"
        }

if __name__ == "__main__":
    # Get port from environment variable (Railway/Render set this)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Starting Bio Age Estimator on port {port}")
    logger.info(f"🌐 Environment: {'Railway' if IS_RAILWAY else 'Local'}")
    
    # Initialize models before starting server
    try:
        model_status = initialize_models()
        
        # Check if at least face detector and deepface are working
        if not model_status['face_detector'] or not model_status['deepface']:
            logger.warning("⚠️ Essential models failed to load, but continuing deployment")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.warning("⚠️ Continuing deployment without models")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    ) 