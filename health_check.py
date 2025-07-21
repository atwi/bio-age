#!/usr/bin/env python3
"""
Health check script for Railway deployment verification
Checks model availability and API functionality
"""

import os
import sys
import time
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_files():
    """Check if model files exist"""
    logger.info("🔍 Checking model files...")
    
    model_paths = [
        "model_saved_tf",
        "./model_saved_tf",
        "FaceAge/model_saved_tf"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found model at: {path}")
            try:
                contents = os.listdir(path)
                logger.info(f"   Contents: {contents}")
                return True
            except Exception as e:
                logger.warning(f"   Error reading contents: {e}")
        else:
            logger.info(f"❌ Model not found at: {path}")
    
    return False

def check_environment():
    """Check environment variables"""
    logger.info("🌍 Checking environment...")
    
    env_vars = {
        'RAILWAY_ENVIRONMENT': os.environ.get('RAILWAY_ENVIRONMENT', 'NOT_SET'),
        'LOAD_HARVARD_MODEL': os.environ.get('LOAD_HARVARD_MODEL', 'NOT_SET'),
        'ENABLE_DEEPFACE': os.environ.get('ENABLE_DEEPFACE', 'NOT_SET'),
        'OPENAI_API_KEY': 'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT_SET'
    }
    
    for var, value in env_vars.items():
        logger.info(f"   {var}: {value}")
    
    return True

def test_local_api():
    """Test local API if available"""
    logger.info("🏥 Testing local API...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Health endpoint working")
            data = response.json()
            logger.info(f"   Status: {data.get('status')}")
            logger.info(f"   Environment: {data.get('environment')}")
        else:
            logger.warning(f"⚠️ Health endpoint returned {response.status_code}")
            return False
        
        # Test API health endpoint
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API health endpoint working")
            data = response.json()
            models = data.get('models', {})
            logger.info(f"   Models: {models}")
        else:
            logger.warning(f"⚠️ API health endpoint returned {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        logger.info("ℹ️ Local API not running (this is normal)")
        return True
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    logger.info("📦 Checking dependencies...")
    
    dependencies = [
        'tensorflow',
        'deepface',
        'mtcnn',
        'opencv-python',
        'pillow',
        'numpy',
        'gdown',
        'openai'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            logger.info(f"✅ {dep}")
        except ImportError:
            logger.warning(f"❌ {dep} - missing")
            missing.append(dep)
    
    if missing:
        logger.warning(f"⚠️ Missing dependencies: {missing}")
        return False
    
    return True

def main():
    """Run all health checks"""
    logger.info("🚀 Starting deployment health checks...")
    
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Local API", test_local_api)
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {name} check...")
        try:
            result = check_func()
            results.append((name, result))
            logger.info(f"{name} check: {'✅ PASS' if result else '❌ FAIL'}")
        except Exception as e:
            logger.error(f"❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📋 Health Check Summary:")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        logger.info("🎉 All health checks passed! Deployment should work.")
        return 0
    else:
        logger.warning("⚠️ Some health checks failed. Review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 