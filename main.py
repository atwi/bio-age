#!/usr/bin/env python3
"""
Main deployment script for Railway/Render
Serves both the FastAPI backend and React Native web frontend
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the API backend and its initialization functions
import api_backend
from api_backend import app as api_app, init_face_detector, load_harvard_model, test_deepface

# Create the main app
app = FastAPI(title="Bio Age Estimator", description="Full-stack deployment")

# Initialize models on startup
def initialize_models():
    """Initialize models on startup"""
    print("Initializing models...")
    
    # Initialize face detector and set the global variable
    if init_face_detector():
        print("✅ Face detector initialized successfully")
    else:
        print("❌ Face detector initialization failed")
    
    # Initialize Harvard model and set the global variable
    if load_harvard_model():
        print("✅ Harvard model loaded successfully") 
    else:
        print("❌ Harvard model not available")
    
    # Test DeepFace
    if test_deepface():
        print("✅ DeepFace test successful")
    else:
        print("❌ DeepFace not available")
    
    print("API server ready!")
    
    # Verify the models are set in api_backend
    print(f"Face detector status: {api_backend.face_detector is not None}")
    print(f"Harvard model status: {api_backend.harvard_model is not None}")

# Mount the API backend under /api
app.mount("/api", api_app)

# Check if React Native web build exists
web_build_path = Path("FaceAgeApp/dist")
if web_build_path.exists():
    # Serve static files (React Native web build)
    app.mount("/static", StaticFiles(directory=str(web_build_path)), name="static")
    
    # Serve the React Native app for all other routes
    @app.get("/{path:path}")
    async def serve_react_app(path: str):
        """Serve the React Native web app"""
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
            "note": "React Native web build not found. Run 'cd FaceAgeApp && npx expo export --platform web' to build the frontend."
        }

# Health check for the main app
@app.get("/health")
async def health_check():
    """Health check for the main deployment"""
    return {
        "status": "healthy",
        "service": "bio-age-estimator",
        "frontend": "available" if web_build_path.exists() else "not_built",
        "backend": "available"
    }

if __name__ == "__main__":
    # Get port from environment variable (Railway/Render set this)
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting Bio Age Estimator on port {port}")
    print(f"📁 Web build path: {web_build_path}")
    print(f"🌐 Frontend available: {web_build_path.exists()}")
    
    # Initialize models before starting server
    initialize_models()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    ) 