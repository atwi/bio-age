# 🧬 TrueAge - AI Age Estimation App

TrueAge uses advanced AI models to estimate your biological and perceived age from facial photos. Built with React Native/Expo, featuring Firebase authentication and powered by Harvard FaceAge, DeepFace, and OpenAI GPT-4o Vision models.

## 🌟 Features

### 🤖 **Multi-Model AI Analysis**
- **Harvard FaceAge**: Research-grade biological age estimation (best for 40+ years)
- **DeepFace**: General-purpose facial age analysis with VGG-Face architecture
- **ChatGPT Vision (GPT-4o)**: Human-like age perception with detailed explanations
- **Mean Age Calculation**: Averaged results across all models

### 📱 **Cross-Platform Mobile App**
- **React Native/Expo**: Works on iOS, Android, and Web
- **Camera Integration**: Take photos directly or upload from gallery
- **Responsive Design**: Optimized for all screen sizes
- **Native-like UI**: Full-screen camera with face-framing guide

### 🔐 **User Authentication & Data**
- **Firebase Authentication**: Google and Apple sign-in
- **User Profiles**: Automatic account creation and management
- **Analysis History**: Save and retrieve past results
- **Cloud Storage**: Secure image and data storage

### 🎨 **Modern UI/UX**
- **UI Kitten Components**: Consistent, professional design
- **Glassmorphism Effects**: Modern visual styling
- **Animated Scanning**: Real-time analysis feedback
- **Beta-Focused Messaging**: Community-oriented for feedback collection

### 📊 **Detailed Analysis**
- **Age Factors Breakdown**: Skin texture, tone, hair, and facial volume analysis
- **Confidence Scoring**: Quality assessment for each detection
- **Face Mesh Visualization**: Advanced facial landmark detection
- **Shareable Results**: Export and share analysis results

## 🚀 Live Demo

**Web App**: [https://your-railway-app.railway.app](https://your-railway-app.railway.app)

**Mobile**: Download the Expo app and scan the QR code from the development server

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+ and npm/yarn
- Expo CLI (`npm install -g @expo/cli`)
- Firebase project with Authentication enabled
- OpenAI API key
- Python backend server (for AI models)

### 1. Clone Repository
```bash
git clone https://github.com/atwi/bio-age.git
cd bio-age/FaceAgeApp
```

### 2. Install Dependencies
```bash
npm install
# or
yarn install
```

### 3. Firebase Configuration
1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Enable Authentication with Google and Apple providers
3. Enable Firestore Database and Storage
4. Copy your Firebase config to `firebase.js`:

```javascript
const firebaseConfig = {
  apiKey: "your-api-key",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.firebasestorage.app",
  messagingSenderId: "your-sender-id",
  appId: "your-app-id"
};
```

### 4. Backend Server Setup
```bash
# In the root directory
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ENABLE_DEEPFACE=true
export LOAD_HARVARD_MODEL=true
export PORT=8001

# Run the Python backend
python api_backend.py
```

### 5. Start Development Server
```bash
# In FaceAgeApp directory
npx expo start

# For web development
npx expo start --web

# For mobile development
npx expo start --tunnel  # Use tunnel for external device testing
```

## 🏗️ Architecture

### Frontend (React Native/Expo)
- **Framework**: React Native with Expo managed workflow
- **UI Library**: UI Kitten for consistent components
- **State Management**: React hooks and context
- **Navigation**: Single-page app with step-based navigation
- **Authentication**: Firebase Auth with Google/Apple providers
- **Database**: Firestore for user data and analysis history

### Backend (Python FastAPI)
- **Framework**: FastAPI for high-performance API
- **AI Models**: 
  - Harvard FaceAge (TensorFlow)
  - DeepFace library
  - OpenAI GPT-4o Vision API
- **Face Detection**: MTCNN for robust face detection
- **Image Processing**: OpenCV and PIL for preprocessing

### Deployment
- **Frontend**: Railway/Vercel for web deployment
- **Mobile**: Expo Application Services (EAS) for app store builds
- **Backend**: Railway for Python API hosting
- **Database**: Firebase (Firestore + Storage)

## 📱 Usage

### 1. **Take or Upload Photo**
- Use the camera button for live photo capture
- Choose from gallery to upload existing photos
- Face-framing guide helps with optimal positioning

### 2. **AI Analysis**
- Automatic face detection and cropping
- Parallel processing across multiple AI models
- Real-time scanning animation during analysis

### 3. **View Results**
- Age estimates from Harvard, DeepFace, and ChatGPT models
- Mean age calculation across all models
- Detailed breakdown of aging factors (premium feature)

### 4. **Account Features** (Beta Users)
- Sign up with Google or Apple
- Automatic saving of analysis history
- Access to detailed aging factor breakdowns
- Community feedback collection

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Backend API
OPENAI_API_KEY=your_openai_api_key
ENABLE_DEEPFACE=true
LOAD_HARVARD_MODEL=true
PORT=8001

# Firebase (configured in firebase.js)
# See Firebase setup section above
```

### App Configuration (app.json)
```json
{
  "expo": {
    "name": "TrueAge",
    "slug": "trueage",
    "platforms": ["ios", "android", "web"],
    "version": "1.0.0"
  }
}
```

## 🧪 Development

### Running Tests
```bash
# Frontend tests
npm test

# Backend tests
pytest
```

### Building for Production
```bash
# Web build
npx expo export:web

# Mobile builds (requires EAS CLI)
eas build --platform ios
eas build --platform android
```

## 🤝 Contributing

We're in beta and actively seeking feedback! Here's how to contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with clear commit messages
4. **Test thoroughly** on multiple platforms
5. **Submit a pull request** with detailed description

### Contribution Areas
- UI/UX improvements
- Additional AI model integrations
- Performance optimizations
- Accessibility enhancements
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Harvard FaceAge Research Team** - For their groundbreaking age estimation model
- **DeepFace Contributors** - For the robust face analysis library
- **OpenAI** - For GPT-4o Vision API capabilities
- **Firebase Team** - For authentication and database services
- **Expo Team** - For the excellent React Native development platform
- **UI Kitten** - For beautiful, consistent UI components

## 🔗 Links

- **Live App**: [https://your-app.railway.app](https://your-app.railway.app)
- **Firebase Console**: [https://console.firebase.google.com](https://console.firebase.google.com)
- **Expo Documentation**: [https://docs.expo.dev](https://docs.expo.dev)
- **React Native**: [https://reactnative.dev](https://reactnative.dev)
- **UI Kitten**: [https://akveo.github.io/react-native-ui-kitten](https://akveo.github.io/react-native-ui-kitten)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/atwi/bio-age/issues)
- **Email**: alexthorburnwinsor@gmail.com
- **Beta Feedback**: We're actively collecting user feedback for improvements!

---

**Built with ❤️ for the Reddit community and age estimation research** 