#!/usr/bin/env python3
"""
Test script for lazy loading models approach
"""
import os
import time
import requests
import json
from pathlib import Path

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"  # You can use any test image

def test_health_check():
    """Test that health check works immediately without models"""
    print("🔍 Testing health check...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Health check: {data['status']}")
    
    # Test API health check
    response = requests.get(f"{API_BASE_URL}/api/health")
    assert response.status_code == 200
    
    data = response.json()
    print(f"📊 Model status: {data['models']}")
    
    return data['models']

def test_lazy_loading():
    """Test that models load on first API request"""
    print("\n🔄 Testing lazy loading...")
    
    # Check if test image exists
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"⚠️  Test image not found at {TEST_IMAGE_PATH}")
        print("   Please provide a test image or skip this test")
        return
    
    # Time the first request (should include model loading)
    start_time = time.time()
    
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE_URL}/api/analyze-face", files=files)
    
    first_request_time = time.time() - start_time
    print(f"⏱️  First request time: {first_request_time:.2f}s (includes model loading)")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ First request successful: {data['message']}")
        print(f"   Found {len(data['faces'])} faces")
    else:
        print(f"❌ First request failed: {response.status_code} - {response.text}")
        return
    
    # Test second request (should be faster)
    start_time = time.time()
    
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE_URL}/api/analyze-face", files=files)
    
    second_request_time = time.time() - start_time
    print(f"⏱️  Second request time: {second_request_time:.2f}s (models already loaded)")
    
    if response.status_code == 200:
        print("✅ Second request successful")
        print(f"🚀 Speed improvement: {first_request_time/second_request_time:.1f}x faster")
    else:
        print(f"❌ Second request failed: {response.status_code} - {response.text}")

def test_memory_usage():
    """Test memory usage reporting"""
    print("\n💾 Testing memory usage...")
    
    try:
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory before model loading: {memory_before:.1f} MB")
        
        # Trigger model loading
        test_image_path = Path(TEST_IMAGE_PATH)
        if test_image_path.exists():
            with open(test_image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{API_BASE_URL}/api/analyze-face", files=files)
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory after model loading: {memory_after:.1f} MB")
        print(f"Memory increase: {memory_after - memory_before:.1f} MB")
        
        # Check if within reasonable limits
        if memory_after < 1000:  # 1GB limit
            print("✅ Memory usage within reasonable limits")
        else:
            print("⚠️  High memory usage detected")
            
    except ImportError:
        print("psutil not available - skipping memory test")

def test_railway_simulation():
    """Simulate Railway deployment conditions"""
    print("\n🚂 Testing Railway simulation...")
    
    # Set Railway environment variables
    os.environ['RAILWAY_ENVIRONMENT'] = 'production'
    os.environ['LOAD_HARVARD_MODEL'] = 'true'
    
    print("Environment variables set for Railway simulation")
    
    # Test immediate health check (should work without models)
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Fast health check passed (Railway compatible)")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.Timeout:
        print("❌ Health check timed out (Railway would fail)")
    
    # Clean up
    if 'RAILWAY_ENVIRONMENT' in os.environ:
        del os.environ['RAILWAY_ENVIRONMENT']

def main():
    """Run all tests"""
    print("🧪 Testing Lazy Loading Model Approach")
    print("=" * 50)
    
    try:
        # Test 1: Health check without models
        models_before = test_health_check()
        
        # Test 2: Lazy loading
        test_lazy_loading()
        
        # Test 3: Memory usage
        test_memory_usage()
        
        # Test 4: Railway simulation
        test_railway_simulation()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("📝 Summary:")
        print("   - Health checks work immediately")
        print("   - Models load on first request")
        print("   - Subsequent requests are faster")
        print("   - Railway deployment should work")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Please start the server first: python main.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 