# Railway Deployment Guide

## 🚀 Deployment Optimizations for Railway

This deployment has been optimized for Railway's resource constraints:

### 🔧 Key Optimizations

1. **Memory Management**
   - Harvard model disabled by default (`LOAD_HARVARD_MODEL=false`)
   - Only essential models loaded (Face detector + DeepFace)
   - Aggressive garbage collection
   - TensorFlow optimized for CPU-only usage

2. **Build Process**
   - Uses `--no-cache-dir` for pip installs
   - Production npm install only
   - Reduced build dependencies

3. **Error Handling**
   - Graceful fallback if models fail to load
   - Detailed logging for debugging
   - Continues deployment even if some models fail

### 🚀 Deployment Steps

1. **Push to Railway:**
   ```bash
   git add .
   git commit -m "Railway optimized deployment"
   git push
   ```

2. **Monitor deployment logs** to see:
   - Model loading status
   - Memory usage
   - Error messages

3. **Test endpoints:**
   - Health check: `https://your-app.railway.app/health`
   - API health: `https://your-app.railway.app/api/health`

### 🐛 Debugging Common Issues

#### Memory Issues
- Harvard model is disabled by default
- Only face detector + DeepFace should load
- If still failing, consider reducing image processing

#### Build Failures
- Check build logs for specific error messages
- Ensure all dependencies are in requirements.txt
- Verify Node.js/npm installation

#### Model Loading Failures
- App should continue without Harvard model
- Face detector + DeepFace are essential
- Check environment variables

### 🎯 Environment Variables

Set these in Railway dashboard:

```bash
TF_CPP_MIN_LOG_LEVEL=3
TF_ENABLE_ONEDNN_OPTS=0
CUDA_VISIBLE_DEVICES=-1
LOAD_HARVARD_MODEL=false
RAILWAY_ENVIRONMENT=production
```

### 📊 Expected Behavior

1. **Startup:** Models load with graceful fallback
2. **Face Detection:** Works with MTCNN
3. **Age Estimation:** Works with DeepFace only
4. **Frontend:** React Native web app serves from `/`
5. **API:** Available at `/api/` endpoints

### 🚨 If Still Failing

1. **Check Railway logs** for specific errors
2. **Test locally** with `LOAD_HARVARD_MODEL=false`
3. **Reduce model complexity** further if needed
4. **Contact Railway support** if memory limits are exceeded

### 📈 Performance Expectations

- **Startup time:** 2-3 minutes (model loading)
- **Memory usage:** <512MB (without Harvard model)
- **Response time:** 2-5 seconds per face analysis
- **Accuracy:** DeepFace only (no Harvard model)

### 🔄 Enabling Harvard Model (Optional)

If Railway deployment is successful, you can try enabling Harvard model:

```bash
LOAD_HARVARD_MODEL=true
```

**Warning:** This may cause memory issues and deployment failures. 