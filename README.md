# 📱 TrueAge

TrueAge uses advanced AI models to estimate your biological age and perceived age from a single facial photo. Upload your selfie to discover insights about your health and aging, powered by state-of-the-art deep learning and facial analysis technology.

## 🌟 Features

- **Dual Age Estimation**: Uses both Harvard FaceAge (used by TrueAge) and DeepFace models for accurate predictions
- **Face Detection**: Automatic face detection using MTCNN
- **Mobile Optimized**: Responsive design for mobile devices
- **Multiple Input Methods**: Upload images or take photos directly
- **Results Download**: Download age estimation results as text files

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourusername/bio-age/main/streamlit_app.py)

## 🛠️ Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bio-age.git
cd bio-age
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file with your configuration
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "ENABLE_DEEPFACE=true" >> .env
echo "LOAD_HARVARD_MODEL=true" >> .env
```

**Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key for ChatGPT age estimation
- `ENABLE_DEEPFACE`: Set to `false` to disable DeepFace (useful for cloud deployments with limited resources)
- `LOAD_HARVARD_MODEL`: Set to `false` to disable Harvard model
- `PORT`: Server port (default: 8000)

5. Run the application:
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment

This app is optimized for Streamlit Cloud deployment:

1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy directly from your GitHub repository

## 🔧 Technical Details

### Models Used

- **Harvard FaceAge (used by TrueAge)**: Research-grade model for precise age estimation
- **DeepFace**: Industry-standard model with robust performance
- **ChatGPT Vision**: AI-powered human-like age perception
- **MTCNN**: Face detection and alignment

### Technology Stack

- **Frontend**: Streamlit
- **Backend**: TensorFlow/Keras 2.13.0
- **Computer Vision**: OpenCV, PIL
- **Face Detection**: MTCNN
- **Age Estimation**: DeepFace 0.0.93

### Requirements

- Python 3.11.3
- TensorFlow 2.13.0
- Compatible with Streamlit Cloud

## 📝 Usage

1. **Upload Image**: Choose an image file (JPG, JPEG, PNG)
2. **Take Photo**: Use your device's camera
3. **View Results**: See age estimates from both models
4. **Download**: Save results as a text file

## 🎯 Accuracy

The app provides dual age estimates:
- Harvard FaceAge (used by TrueAge): Research-grade precision
- DeepFace: Robust across different demographics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Harvard FaceAge research team (used by TrueAge)
- DeepFace library contributors
- Streamlit team for the amazing framework

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [DeepFace Library](https://github.com/serengil/deepface)
- [TensorFlow](https://tensorflow.org) 