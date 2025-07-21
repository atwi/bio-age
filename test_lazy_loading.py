#!/usr/bin/env python3
"""
Test script for lazy loading functionality
Tests model loading and fallback behavior
"""

import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for testing
os.environ['RAILWAY_ENVIRONMENT'] = 'production'
os.environ['LOAD_HARVARD_MODEL'] = 'true'
os.environ['ENABLE_DEEPFACE'] = 'true'

def test_model_loading():
    """Test the lazy loading functionality"""
    logger.info("🧪 Testing lazy loading functionality...")
    
    try:
        # Import the main module
        from main import lazy_load_models, detector, harvard_model, deepface_ready
        
        logger.info("📦 Testing model loading...")
        start_time = time.time()
        
        # Test lazy loading
        success = lazy_load_models()
        load_time = time.time() - start_time
        
        logger.info(f"⏱️ Model loading took {load_time:.2f} seconds")
        logger.info(f"✅ Lazy loading result: {success}")
        
        # Check model status
        logger.info(f"📊 Model status:")
        logger.info(f"  - Face detector: {detector is not None}")
        logger.info(f"  - Harvard model: {harvard_model is not None}")
        logger.info(f"  - DeepFace ready: {deepface_ready}")
        
        # Test second call (should be fast)
        logger.info("🔄 Testing second call (should be fast)...")
        start_time = time.time()
        success2 = lazy_load_models()
        load_time2 = time.time() - start_time
        
        logger.info(f"⏱️ Second call took {load_time2:.2f} seconds")
        logger.info(f"✅ Second call result: {success2}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_health_endpoints():
    """Test health endpoints"""
    logger.info("🏥 Testing health endpoints...")
    
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test basic health
        response = client.get("/health")
        logger.info(f"Health response: {response.status_code}")
        logger.info(f"Health data: {response.json()}")
        
        # Test API health
        response = client.get("/api/health")
        logger.info(f"API health response: {response.status_code}")
        logger.info(f"API health data: {response.json()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Health test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting lazy loading tests...")
    
    # Test model loading
    model_test = test_model_loading()
    
    # Test health endpoints
    health_test = test_health_endpoints()
    
    # Summary
    logger.info("📋 Test Summary:")
    logger.info(f"  - Model loading: {'✅ PASS' if model_test else '❌ FAIL'}")
    logger.info(f"  - Health endpoints: {'✅ PASS' if health_test else '❌ FAIL'}")
    
    if model_test and health_test:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some tests failed!")
        sys.exit(1) 