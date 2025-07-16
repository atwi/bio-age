#!/bin/bash
echo "🏗️  Building React Native Web App..."

# Navigate to the React Native app directory
cd FaceAgeApp

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the web version
echo "🌐 Building web version..."
npx expo export --platform web

# Move the build to the correct location
echo "📁 Moving build files..."
if [ -d "dist" ]; then
    echo "✅ Web build created successfully at FaceAgeApp/dist"
else
    echo "❌ Web build failed"
    exit 1
fi

cd ..
echo "🎉 Build complete! Ready for deployment." 