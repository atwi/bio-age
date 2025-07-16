# Railway Deployment Guide

## Overview
This guide covers deploying the Bio Age Estimator to Railway with the new **lazy loading** approach that resolves deployment crashes.

## ✅ New Lazy Loading Approach

### Key Features
- **Models load on first API request** (not during startup)
- **Sequential loading** with garbage collection between models
- **Fast startup** that passes Railway health checks
- **Both Harvard + DeepFace models enabled** by default
- **Memory-optimized** with aggressive cleanup

### Why This Works
The previous deployment failures were caused by:
1. **Startup timeouts** - loading all models during startup
2. **Memory spikes** - concurrent model loading 
3. **Health check failures** - long initialization times

The lazy loading approach solves these issues by:
1. **Immediate startup** - health checks pass instantly
2. **Controlled memory usage** - sequential loading with cleanup
3. **Better user experience** - first request loads models, subsequent requests are fast

## 🚀 Deployment Steps

### 1. Connect Repository
1. Go to [Railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `bio-age` repository
4. Choose "Deploy Now"

### 2. Environment Variables
Railway will automatically set these from `railway.json`:
```bash
RAILWAY_ENVIRONMENT=production
LOAD_HARVARD_MODEL=true          # ✅ Now enabled by default
TF_CPP_MIN_LOG_LEVEL=3
TF_ENABLE_ONEDNN_OPTS=0
CUDA_VISIBLE_DEVICES=-1
TF_FORCE_GPU_ALLOW_GROWTH=true
PYTHONUNBUFFERED=1
NODE_ENV=production
```

### 3. Build Process
The build will:
1. Install Python dependencies
2. Build React Native web frontend
3. Create deployment package

### 4. First Request
- **Health checks** pass immediately
- **First API request** triggers model loading (may take 30-60 seconds)
- **Subsequent requests** are fast (<2 seconds)

## 📊 Testing Your Deployment

### Local Testing
```bash
# Start the server
python main.py

# Run tests
python test_lazy_loading.py
```

### Railway Testing
```bash
# Test health check (should be instant)
curl https://your-app.railway.app/health

# Test API health (shows model status)
curl https://your-app.railway.app/api/health

# Test face analysis (triggers model loading on first request)
curl -X POST -F "file=@test_image.jpg" https://your-app.railway.app/api/analyze-face
```

## 🔧 Configuration Options

### Model Selection
```bash
# Enable/disable Harvard model
LOAD_HARVARD_MODEL=true   # Default: true

# Railway environment detection
RAILWAY_ENVIRONMENT=production
```

### Memory Management
```bash
# TensorFlow optimizations
TF_CPP_MIN_LOG_LEVEL=3
CUDA_VISIBLE_DEVICES=-1
TF_FORCE_GPU_ALLOW_GROWTH=true
```

## 🚨 Troubleshooting

### If Deployment Still Fails

1. **Check logs** in Railway dashboard
2. **Verify build command** in `railway.json`
3. **Test locally** with Railway environment variables:
   ```bash
   RAILWAY_ENVIRONMENT=production LOAD_HARVARD_MODEL=true python main.py
   ```

### If Models Fail to Load

1. **Check first request logs** - models load on first API call
2. **Verify model files** are included in build
3. **Test with single model**:
   ```bash
   LOAD_HARVARD_MODEL=false python main.py
   ```

### Memory Issues

1. **Monitor memory usage** in Railway dashboard
2. **Check garbage collection** in logs
3. **Sequential loading** prevents memory spikes

## 📈 Performance Expectations

### Startup Time
- **Health checks**: <1 second ✅
- **First API request**: 30-60 seconds (model loading)
- **Subsequent requests**: <2 seconds ✅

### Memory Usage
- **Startup**: ~100MB
- **After model loading**: ~800MB
- **Peak during loading**: ~1GB (briefly)

## 🎯 Model Comparison

| Model | Size | Accuracy | Use Case |
|-------|------|----------|----------|
| **Harvard FaceAge** | 85MB | High for 60+ | Elderly subjects |
| **DeepFace** | 539MB | Good general | All ages (bias toward 40-50) |
| **MTCNN** | 5MB | Face detection | Detection only |

## 📱 Frontend Integration

The React Native app will automatically work with the lazy loading:
- **Health checks** work immediately
- **First photo analysis** may show longer loading time
- **Subsequent photos** process quickly
- **Error handling** for model loading failures

## 🔄 Updates and Maintenance

### Updating Models
1. Models are loaded from `FaceAge/models/` directory
2. Automatic fallback if models fail to load
3. Graceful degradation with error messages

### Monitoring
1. **Railway dashboard** for memory/CPU usage
2. **Health check endpoint** for model status
3. **API logs** for performance monitoring

## 📞 Support

If you encounter issues:
1. Check the **Railway logs** for specific error messages
2. Test **locally** with Railway environment variables
3. Verify **model files** are present in the build
4. Check **memory usage** doesn't exceed limits

The lazy loading approach should resolve most deployment issues while maintaining full functionality with both models enabled. 