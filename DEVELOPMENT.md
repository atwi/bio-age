# Development Guide

## 🚀 Quick Start with Hot Reloading

### Option 1: Use the Development Script (Recommended)
```bash
./dev-server.sh
```

This will:
- Start the backend API server on `http://localhost:8001`
- Start the frontend development server on `http://localhost:19006`
- Enable hot reloading for both frontend and backend changes

### Option 2: Manual Setup

#### Backend (API Server)
```bash
# Terminal 1
source venv/bin/activate
export DEVELOPMENT=true
python main.py
```

#### Frontend (React Native Web)
```bash
# Terminal 2
cd FaceAgeApp
npm run web
```

## 🔄 Hot Reloading Features

### Frontend Changes
- **React components**: Changes appear immediately
- **Styling**: CSS/theme changes reload automatically
- **Navigation**: Route changes update instantly

### Backend Changes
- **API endpoints**: Restart backend to see changes
- **Model logic**: Restart backend for model updates
- **Configuration**: Environment variable changes require restart

## 🌐 Access Points

- **Frontend (Development)**: http://localhost:19006
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/api/health

## 🛠️ Development Workflow

1. **Start development servers**: `./dev-server.sh`
2. **Make frontend changes**: Edit files in `FaceAgeApp/`
3. **See changes immediately**: Browser updates automatically
4. **Make backend changes**: Edit `main.py` and restart backend
5. **Test API endpoints**: Use http://localhost:8001/docs

## 📁 Project Structure

```
bio-age/
├── main.py                 # Backend API server
├── dev-server.sh          # Development startup script
├── FaceAgeApp/            # Frontend React Native app
│   ├── App.js             # Main app component
│   ├── components/        # React components
│   └── web-build/         # Production build (auto-generated)
└── requirements.txt       # Python dependencies
```

## 🚨 Troubleshooting

### Port Conflicts
The development script will ask you how to handle port conflicts:
1. Stop existing process
2. Use different port
3. Handle manually

### Frontend Not Loading
- Check if Expo dev server is running on port 19006
- Try refreshing the browser
- Check browser console for errors

### Backend Not Responding
- Check if Python server is running on port 8001
- Verify virtual environment is activated
- Check terminal for error messages

## 🏭 Production vs Development

- **Development**: Frontend on Expo dev server (hot reloading)
- **Production**: Frontend served as static files by backend

The system automatically detects which mode to use based on the `DEVELOPMENT` environment variable. 