#!/usr/bin/env python3
"""
Build script for downloading and setting up models
Run this during Railway build phase
"""

import os
import sys
import gdown
import zipfile
import shutil
from pathlib import Path

def download_harvard_model():
    """Download and extract Harvard model"""
    print("📥 Starting Harvard model download...")
    
    # Model URLs and paths
    MODEL_URL = "https://drive.google.com/uc?id=12wNpYBz3j5mP9mt6S_ZH4k0sI6dVDeVV"
    MODEL_ZIP = "model_saved_tf.zip"
    MODEL_DIR = "model_saved_tf"
    
    try:
        # Download model
        print(f"📥 Downloading from: {MODEL_URL}")
        gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)
        
        if not os.path.exists(MODEL_ZIP):
            print("❌ Download failed - zip file not found")
            return False
            
        print(f"📦 Extracting {MODEL_ZIP}...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Clean up zip file
        os.remove(MODEL_ZIP)
        
        # Verify extraction
        if os.path.exists(MODEL_DIR):
            print(f"✅ Model extracted to {MODEL_DIR}")
            print(f"📁 Model directory contents: {os.listdir(MODEL_DIR)}")
            return True
        else:
            print(f"❌ Model directory {MODEL_DIR} not found after extraction")
            return False
            
    except Exception as e:
        print(f"❌ Error during model download: {e}")
        return False

def main():
    """Main build function"""
    print("🚀 Starting model build process...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download Harvard model
    success = download_harvard_model()
    
    if success:
        print("✅ All models downloaded successfully!")
        return 0
    else:
        print("❌ Model download failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 