#!/bin/bash

# Development Server Script for Bio Age Project
# This script runs both backend and frontend with hot reloading

echo "🚀 Starting Bio Age Development Server..."
echo "📁 Backend: http://localhost:8001"
echo "🌐 Frontend: http://localhost:19006 (Expo Dev Server)"

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down development servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if port 8001 is in use
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8001 is already in use."
    echo "Would you like me to:"
    echo "1) Stop the existing process"
    echo "2) Use a different port"
    echo "3) You handle it manually"
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            echo "🛑 Stopping existing process on port 8001..."
            lsof -ti:8001 | xargs kill -9
            sleep 2
            ;;
        2)
            echo "Please specify a different port:"
            read -p "Enter port number: " new_port
            export PORT=$new_port
            ;;
        3)
            echo "Please handle the port conflict manually and restart this script."
            exit 1
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Start backend server with auto-reload
echo "🔧 Starting backend server (auto-reload enabled)..."
cd /Users/alext/Documents/GitHub/bio-age
source venv/bin/activate
export DEVELOPMENT=true
# Use uvicorn reload so code changes are picked up without manual restarts
uvicorn main:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend started successfully
if ! curl -s http://localhost:8001/api/health > /dev/null; then
    echo "❌ Backend failed to start. Check the logs above."
    cleanup
fi

echo "✅ Backend server started successfully!"

# Start frontend development server
echo "🎨 Starting frontend development server..."
cd FaceAgeApp
npm run web &
FRONTEND_PID=$!

echo "🎉 Development servers are running!"
echo ""
echo "📱 Backend API: http://localhost:8001"
echo "🌐 Frontend: http://localhost:19006"
echo "📊 Health Check: http://localhost:8001/api/health"
echo ""
echo "💡 Hot reloading is enabled - backend changes auto-restart; frontend uses HMR."
echo "🛑 Press Ctrl+C to stop both servers"

# Wait for user to stop
wait 