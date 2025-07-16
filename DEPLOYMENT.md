# 🚀 Bio Age Estimator - Deployment Guide

Deploy your facial age estimation app online so users can access it via web browsers on mobile and desktop!

## 📋 Prerequisites

- Git repository with your code
- GitHub/GitLab account
- Basic terminal/command line knowledge

## 🏗️ Build the Frontend

First, build the React Native web version:

```bash
# Make the build script executable
chmod +x build-web.sh

# Build the web app
./build-web.sh
```

## 🎯 Deployment Options

### Option 1: Railway (Recommended) ⭐

**Best for**: Full-stack deployment with automatic builds

1. **Sign up**: Go to [Railway.app](https://railway.app) and sign up with GitHub
2. **Connect Repository**: Click "Deploy from GitHub repo" and select your repository
3. **Configure**: Railway will auto-detect your Python app
4. **Deploy**: Click "Deploy Now"

**Pros**: 
- ✅ Automatic builds
- ✅ Free tier available
- ✅ Handles both frontend and backend
- ✅ Automatic HTTPS

**Cons**: 
- ⚠️ Usage limits on free tier

### Option 2: Render ⭐

**Best for**: Reliable hosting with good free tier

1. **Sign up**: Go to [Render.com](https://render.com) and sign up with GitHub
2. **Create Web Service**: 
   - Choose "Web Service"
   - Connect your GitHub repository
   - Build Command: `./build-web.sh && pip install -r requirements.txt`
   - Start Command: `python main.py`
3. **Deploy**: Click "Create Web Service"

**Pros**:
- ✅ Generous free tier
- ✅ Automatic SSL
- ✅ Easy database integration
- ✅ Good performance

**Cons**:
- ⚠️ Cold starts on free tier

### Option 3: Heroku

**Best for**: Enterprise-grade hosting

1. **Install Heroku CLI**: Download from [heroku.com](https://heroku.com)
2. **Login**: `heroku login`
3. **Create app**: `heroku create your-app-name`
4. **Deploy**: 
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

**Pros**:
- ✅ Mature platform
- ✅ Many add-ons
- ✅ Enterprise features

**Cons**:
- ❌ No free tier anymore
- ❌ More expensive

### Option 4: Google Cloud Run

**Best for**: Scalable serverless deployment

1. **Enable Google Cloud Run**: In Google Cloud Console
2. **Build and deploy**:
   ```bash
   # Build the container
   docker build -t bio-age-estimator .
   
   # Deploy to Cloud Run
   gcloud run deploy bio-age-estimator \
     --image bio-age-estimator \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

**Pros**:
- ✅ Serverless scaling
- ✅ Pay per request
- ✅ Google infrastructure

**Cons**:
- ⚠️ Requires Docker knowledge
- ⚠️ Cold starts

### Option 5: AWS Lambda (Serverless)

**Best for**: Ultra-scalable serverless deployment

1. **Install Serverless Framework**: `npm install -g serverless`
2. **Configure AWS credentials**
3. **Deploy**: `serverless deploy`

**Pros**:
- ✅ Infinite scaling
- ✅ Pay per request
- ✅ AWS ecosystem

**Cons**:
- ⚠️ Complex setup
- ⚠️ Cold starts
- ⚠️ Model loading challenges

## 🔧 Configuration

### Environment Variables

Set these in your hosting platform:

```bash
# Python settings
PYTHONPATH=/app
PORT=8000

# Optional: Model optimization
TF_CPP_MIN_LOG_LEVEL=3
```

### Domain Setup

Most platforms provide a free subdomain like:
- Railway: `your-app.railway.app`
- Render: `your-app.onrender.com`
- Heroku: `your-app.herokuapp.com`

For custom domains, add your domain in the platform's dashboard.

## 🎯 Testing Your Deployment

After deployment, test these endpoints:

1. **Health Check**: `https://your-app.com/health`
2. **API Health**: `https://your-app.com/api/health`
3. **Web App**: `https://your-app.com/`
4. **API Docs**: `https://your-app.com/api/docs`

## 📱 Mobile Access

Your deployed app will work on:
- ✅ Mobile web browsers (iOS Safari, Android Chrome)
- ✅ Desktop browsers
- ✅ Progressive Web App (PWA) features
- ✅ Share directly via URL

## 🔍 Monitoring

### Basic Monitoring

Most platforms provide:
- Request logs
- Error tracking
- Performance metrics
- Uptime monitoring

### Advanced Monitoring

For production apps, consider:
- **Sentry**: Error tracking
- **LogRocket**: Session replay
- **New Relic**: APM
- **DataDog**: Full monitoring

## 💰 Cost Estimates

### Free Tiers:
- **Railway**: $5/month after free tier
- **Render**: Free tier available
- **Vercel**: Free tier for personal projects
- **Netlify**: Free tier for static sites

### Paid Tiers:
- **Railway**: $5-20/month
- **Render**: $7-25/month
- **Heroku**: $7-25/month
- **Google Cloud**: Pay-per-use

## 🚀 Quick Start (Railway)

1. **Build the app**:
   ```bash
   ./build-web.sh
   ```

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

3. **Deploy on Railway**:
   - Go to [Railway.app](https://railway.app)
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Click "Deploy Now"

4. **Done!** Your app will be live at `https://your-app.railway.app`

## 🔧 Troubleshooting

### Common Issues:

**Build Fails**:
```bash
# Check build logs
# Ensure all dependencies are in requirements.txt
# Verify Node.js version compatibility
```

**Model Loading Fails**:
```bash
# Check if model files are in repository
# Verify model path in code
# Ensure sufficient memory allocation
```

**API Errors**:
```bash
# Check API logs
# Verify CORS settings
# Test API endpoints directly
```

## 📞 Support

If you need help:
1. Check the deployment logs
2. Test locally first
3. Verify all files are committed to Git
4. Check platform-specific documentation

## 🎉 Next Steps

After deployment:
1. Share your app URL with users
2. Monitor usage and performance
3. Set up custom domain (optional)
4. Add analytics tracking
5. Plan for App Store submission

Your app is now accessible to anyone with the URL! 🌐 