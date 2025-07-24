import React, { useState, useEffect, Fragment, useRef } from 'react';
import {
  StyleSheet,
  Image,
  Alert,
  ScrollView,
  ActivityIndicator,
  Dimensions,
  Platform,
  SafeAreaView,
  Animated,
  View,
  Share,
  TouchableOpacity,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import * as Sharing from 'expo-sharing';
import { Ionicons } from '@expo/vector-icons';
import Constants from 'expo-constants';
import { LinearGradient } from 'expo-linear-gradient';

// UI Kitten imports
import * as eva from '@eva-design/eva';
import { EvaIconsPack } from '@ui-kitten/eva-icons';
import {
  ApplicationProvider,
  Layout,
  Text,
  Button,
  Card,
  Avatar,
  Divider,
  Icon,
  Spinner,
  IconRegistry,
  Modal,
  Tooltip,
} from '@ui-kitten/components';

const { width, height } = Dimensions.get('window');

// API Configuration
const getApiBaseUrl = () => {
  if (typeof window !== 'undefined' && window.location) {
    // Web environment
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8001/api';
    } else {
      return '/api';
    }
  } else if (Constants.manifest?.debuggerHost) {
    // Expo Go (mobile) - get Metro Bundler IP
    const debuggerHost = Constants.manifest.debuggerHost.split(':')[0];
    return `http://${debuggerHost}:8001/api`;
  } else {
    // Fallback
    return 'http://192.168.96.123:8001/api';
  }
};

const API_BASE_URL = getApiBaseUrl();

// Custom icons for UI Kitten
const CameraIcon = (props) => (
  <Icon {...props} name='camera-outline'/>
);

const ImageIcon = (props) => (
  <Icon {...props} name='image-outline'/>
);

const AnalyticsIcon = (props) => (
  <Icon {...props} name='activity-outline'/>
);

const RefreshIcon = (props) => (
  <Icon {...props} name='refresh-outline'/>
);

const ArrowBackIcon = (props) => (
  <Icon {...props} name='arrow-back-outline'/>
);

const ShareIcon = (props) => (
  <Icon {...props} name='share-outline'/>
);

const InfoIcon = (props) => (
  <Icon {...props} name='info-outline'/>
);

const ANALYZE_IMAGE_SIZE = Math.min(width * 0.8, 320);

const AGE_FACTOR_ICONS = {
  skin_texture: '🧬',
  skin_tone: '🎨',
  hair: '💇',
  facial_volume: '💉',
};
const AGE_FACTOR_LABELS = {
  skin_texture: 'Skin Texture',
  skin_tone: 'Skin Tone',
  hair: 'Hair',
  facial_volume: 'Facial Volume',
};

// At the top, add icons/labels for models:
const MODEL_ICONS = {
  harvard: '🎓',
  deepface: '🤖',
  chatgpt: '💬',
  mean: '📊',
};
const MODEL_LABELS = {
  harvard: 'Harvard',
  deepface: 'DeepFace',
  chatgpt: 'ChatGPT',
  mean: 'Mean Age',
};

const MODEL_DESCRIPTIONS = {
  harvard: 'Best for clinical/biological age estimation, people >40',
  deepface: 'Best for general face age estimation',
  chatgpt: 'Best for human-like age perception',
};

function AppContent() {
  const [currentStep, setCurrentStep] = useState(1); // 1: Upload, 2: Analyzing, 3: Results
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [showWebCamera, setShowWebCamera] = useState(false);
  const [webCameraStream, setWebCameraStream] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  // Animated values for scanning effect
  const scanLinePosition = useRef(new Animated.Value(0)).current;
  const scanLineOpacity = useRef(new Animated.Value(0.3)).current;
  const [infoVisible, setInfoVisible] = useState(false);
  const [harvardTooltipVisible, setHarvardTooltipVisible] = useState(false);
  const [deepfaceTooltipVisible, setDeepfaceTooltipVisible] = useState(false);
  const [chatgptTooltipVisible, setChatgptTooltipVisible] = useState(false);
  const [showApiTooltip, setShowApiTooltip] = useState(false);
  const [expandedModel, setExpandedModel] = useState(null);

  // Request permissions on mount
  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'web') {
        const cameraPerm = await ImagePicker.requestCameraPermissionsAsync();
        const mediaPerm = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (cameraPerm.status !== 'granted' || mediaPerm.status !== 'granted') {
          Alert.alert('Permissions Required', 'Camera and photo library permissions are required to use this app.');
        }
      }
    })();
  }, []);

  // Check API health
  useEffect(() => {
    checkApiHealth();
  }, []);

  // Start scanning animation when in step 2
  useEffect(() => {
    if (currentStep === 2) {
      // Start the scanning animation
      const startScanAnimation = () => {
        // Reset position
        scanLinePosition.setValue(0);
        
        // Create looping animation
        Animated.loop(
          Animated.sequence([
            Animated.timing(scanLinePosition, {
              toValue: 1,
              duration: 2000,
              useNativeDriver: false,
            }),
            Animated.timing(scanLinePosition, {
              toValue: 0,
              duration: 2000,
              useNativeDriver: false,
            }),
          ])
        ).start();

        // Pulse opacity
        Animated.loop(
          Animated.sequence([
            Animated.timing(scanLineOpacity, {
              toValue: 0.8,
              duration: 1000,
              useNativeDriver: false,
            }),
            Animated.timing(scanLineOpacity, {
              toValue: 0.3,
              duration: 1000,
              useNativeDriver: false,
            }),
          ])
        ).start();
      };

      startScanAnimation();
    }
  }, [currentStep]);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setApiHealth(data);
      } else {
        setApiHealth({ status: 'error', message: 'API not responding' });
      }
    } catch (error) {
      console.error('Health check error:', error);
      setApiHealth({ status: 'error', message: 'Network error' });
    }
  };

  const resizeImage = async (uri) => {
    try {
      const result = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 800 } }],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
      );
      return result;
    } catch (error) {
      console.error('Resize error:', error);
      return { uri };
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });

      if (!result.canceled) {
        const compressedImage = await resizeImage(result.assets[0].uri);
        setSelectedImage(compressedImage);
        setCurrentStep(2); // Move to analyzing step
        analyzeFace(compressedImage);
      }
    } catch (error) {
      console.error('Pick image error:', error);
      Alert.alert('Error', 'Failed to select image: ' + error.message);
    }
  };

  const takePhoto = async () => {
    try {
      if (Platform.OS === 'web') {
        setShowWebCamera(true);
      } else {
        const result = await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: true,
          aspect: [4, 3],
          quality: 1,
        });

        if (!result.canceled) {
          const compressedImage = await resizeImage(result.assets[0].uri);
          setSelectedImage(compressedImage);
          setCurrentStep(2); // Move to analyzing step
          analyzeFace(compressedImage);
        }
      }
    } catch (error) {
      console.error('Camera error:', error);
      Alert.alert('Error', 'Failed to take photo: ' + error.message);
    }
  };

  const startWebCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' },
        audio: false 
      });
      setWebCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Web camera error:', error);
      Alert.alert('Error', 'Failed to access camera: ' + error.message);
      setShowWebCamera(false);
    }
  };

  const captureWebPhoto = async () => {
    try {
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        context.drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
          const imageUri = URL.createObjectURL(blob);
          const compressedImage = await resizeImage(imageUri);
          setSelectedImage(compressedImage);
          closeWebCamera();
          setCurrentStep(2); // Move to analyzing step
          analyzeFace(compressedImage);
        }, 'image/jpeg', 0.8);
      }
    } catch (error) {
      console.error('Capture error:', error);
      Alert.alert('Error', 'Failed to capture photo: ' + error.message);
    }
  };

  const closeWebCamera = () => {
    if (webCameraStream) {
      webCameraStream.getTracks().forEach(track => track.stop());
      setWebCameraStream(null);
    }
    setShowWebCamera(false);
  };

  useEffect(() => {
    if (showWebCamera && Platform.OS === 'web') {
      startWebCamera();
    }
    return () => {
      if (webCameraStream) {
        webCameraStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [showWebCamera]);

  const analyzeFace = async (imageToAnalyze = selectedImage) => {
    if (!imageToAnalyze) return;

    setLoading(true);
    try {
      const formData = new FormData();
      
      if (typeof imageToAnalyze.uri === 'string' && 
          (imageToAnalyze.uri.startsWith('data:') || imageToAnalyze.uri.startsWith('blob:'))) {
        const response = await fetch(imageToAnalyze.uri);
        const blob = await response.blob();
        formData.append('file', blob, 'image.jpg');
      } else {
        const fileInfo = {
          uri: imageToAnalyze.uri,
          type: 'image/jpeg',
          name: 'image.jpg',
        };
        
        formData.append('file', fileInfo);
      }

      console.log('Sending request to:', `${API_BASE_URL}/analyze-face`);

      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout
      
      const response = await fetch(`${API_BASE_URL}/analyze-face`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      console.log('Response status:', response.status);

      if (response.ok) {
        const data = await response.json();
        console.log('Analysis results:', data);
        // Log ChatGPT debug info for each face
        if (data.faces) {
          data.faces.forEach((face, i) => {
            console.log(`Face ${i}: ChatGPT raw response:`, face.chatgpt_raw_response);
            console.log(`Face ${i}: ChatGPT fallback text:`, face.chatgpt_fallback_text);
            console.log(`Face ${i}: ChatGPT error:`, face.chatgpt_error);
          });
        }
        setResults(data);
        
        // Add minimum delay to show scanning animation (reduced for Railway)
        await new Promise(resolve => setTimeout(resolve, 2000));
        setCurrentStep(3); // Move to results step
      } else {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        Alert.alert('Error', `Analysis failed: ${response.status}`);
        setCurrentStep(1); // Go back to upload step
      }
    } catch (error) {
      console.error('Network error:', error);
      let errorMessage = 'Network error occurred';
      
      if (error.name === 'AbortError') {
        errorMessage = 'Request timed out. The server is taking too long to respond. Please try again.';
      } else if (error.message.includes('Failed to fetch')) {
        errorMessage = 'Unable to connect to the server. Please check your internet connection and try again.';
      } else {
        errorMessage = `Network error: ${error.message}`;
      }
      
      Alert.alert('Error', errorMessage);
      setCurrentStep(1); // Go back to upload step
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedImage(null);
    setResults(null);
    setCurrentStep(1);
    closeWebCamera();
  };

  const shareResults = async () => {
    if (!results || !results.faces || results.faces.length === 0) {
      Alert.alert('Error', 'No results to share');
      return;
    }

    const shareText = generateShareText();

    try {
      if (Platform.OS === 'ios' || Platform.OS === 'android') {
        // Use React Native's Share API for mobile platforms
        await Share.share({
          message: shareText,
          title: 'My Age Analysis Results',
        });
      } else if (Platform.OS === 'web' && navigator.share) {
        // Web Share API is available
        await navigator.share({
          title: 'My Age Analysis Results',
          text: shareText,
        });
      } else if (Platform.OS === 'web' && navigator.clipboard) {
        // Fallback to clipboard on web
        await navigator.clipboard.writeText(shareText);
        Alert.alert('Copied!', 'Results copied to clipboard. You can now paste them anywhere!');
      } else {
        // Final fallback - show alert with copy option
        Alert.alert('Share Results', shareText, [
          { text: 'Copy to Clipboard', onPress: () => copyToClipboard(shareText) },
          { text: 'Cancel', style: 'cancel' }
        ]);
      }
    } catch (error) {
      console.error('Share error:', error);
      // Fallback to clipboard copy
      try {
        await copyToClipboard(shareText);
        Alert.alert('Copied!', 'Results copied to clipboard since sharing failed.');
      } catch (clipboardError) {
        Alert.alert('Share Results', shareText, [
          { text: 'OK', style: 'default' }
        ]);
      }
    }
  };

  const generateShareText = () => {
    if (!results || !results.faces) return '';
    
    const validFaces = results.faces.filter(face => face.confidence >= 0.9);
    if (validFaces.length === 0) return 'No clear faces detected in the analysis.';

    let shareText = '🎯 My Age Analysis Results:\n\n';
    
    validFaces.forEach((face, index) => {
      shareText += `Face ${index + 1}:\n`;
      if (face.age_harvard !== null && face.age_harvard !== undefined) {
        shareText += `🎯 Harvard Model: ${face.age_harvard.toFixed(1)} years\n`;
      }
      if (face.age_deepface !== null && face.age_deepface !== undefined) {
        shareText += `🤖 DeepFace Model: ${face.age_deepface.toFixed(1)} years\n`;
      }
      shareText += `📊 Confidence: ${(face.confidence * 100).toFixed(1)}%\n\n`;
    });

    shareText += 'Try the Bio Age Estimator app yourself! 📱';
    return shareText;
  };

  const copyToClipboard = async (text) => {
    try {
      if (Platform.OS === 'web') {
        await navigator.clipboard.writeText(text);
      } else {
        // For mobile, we could use expo-clipboard if needed
        console.log('Clipboard text:', text);
      }
      Alert.alert('Copied!', 'Results copied to clipboard!');
    } catch (error) {
      console.error('Clipboard error:', error);
      Alert.alert('Copy Failed', 'Unable to copy to clipboard');
    }
  };

  const getResponsiveHeight = () => {
    if (typeof window !== 'undefined' && window.innerHeight) {
      return window.innerHeight;
    }
    return height;
  };

  // Web camera view for browsers
  if (showWebCamera && Platform.OS === 'web') {
    return (
      <SafeAreaView style={styles.container}>
        <Layout style={styles.fullScreen}>
          <Card style={styles.webCameraCard}>
            <Text category='h6' style={styles.webCameraTitle}>📷 Take Photo</Text>
            <Layout style={styles.webCameraContainer}>
              <video
                ref={videoRef}
                style={styles.video}
                autoPlay
                playsInline
                muted
              />
              <canvas
                ref={canvasRef}
                style={{ display: 'none' }}
              />
            </Layout>
            <Layout style={styles.webCameraControls}>
              <Button
                style={styles.webCameraButton}
                onPress={captureWebPhoto}
                accessoryLeft={CameraIcon}
              >
                Capture
              </Button>
              <Button
                style={styles.webCameraButton}
                onPress={closeWebCamera}
                status='basic'
              >
                Cancel
              </Button>
            </Layout>
          </Card>
        </Layout>
      </SafeAreaView>
    );
  }

  const modelDetails = [
    {
      key: 'harvard',
      icon: '🧬',
      name: 'Harvard FaceAge Model',
      summary: 'A research-grade deep learning model for biological age estimation from facial images.',
      details: (
        <>
          <b>Training:</b> Trained on the Harvard FaceAge dataset, with over 56,000 images from clinical and public sources, annotated for biological age.{"\n"}
          <b>Strengths:</b> Designed for medical/biological age, robust to some real-world variation.{"\n"}
          <b>Weaknesses:</b> Systematically inaccurate in people under 40; not trained for cosmetic age.{"\n"}
          <b>Reference:</b> <a href="https://github.com/AIM-Harvard/FaceAge" target="_blank">Harvard FaceAge GitHub</a>{"\n"}
        </>
      )
    },
    {
      key: 'deepface',
      icon: '🤖',
      name: 'DeepFace',
      summary: 'A popular open-source face analysis library using deep learning.',
      details: (
        <>
          <b>Training:</b> Uses VGG-Face, trained on 2.6 million images from 2,622 identities.{"\n"}
          <b>Strengths:</b> Fast, works on many faces, easy to use.{"\n"}
          <b>Weaknesses:</b> Chronological age only, less accurate for older/younger extremes.{"\n"}
          <b>Reference:</b> <a href="https://github.com/serengil/deepface" target="_blank">DeepFace GitHub</a>
        </>
      )
    },
    {
      key: 'gpt',
      icon: '💬',
      name: 'ChatGPT Vision',
      summary: 'A general-purpose vision-language model (GPT-4o) that estimates age from images and text.',
      details: (
        <>
          <b>Training:</b> Trained on a broad mix of web images and text, not specifically for age estimation.{"\n"}
          <b>Strengths:</b> Can explain reasoning, flexible, works on a wide range of images.{"\n"}
          <b>Weaknesses:</b> Not a specialist, may hallucinate or over-interpret features, not always accurate.{"\n"}
          <b>Reference:</b> <a href="https://platform.openai.com/docs/guides/vision" target="_blank">OpenAI GPT-4o</a>
        </>
      )
    }
  ];

  const renderInfoModal = () => {
    const content = (
      <Layout style={styles.infoModalCard}>
        <TouchableOpacity
          style={styles.closeIconContainer}
          onPress={() => setInfoVisible(false)}
          activeOpacity={0.7}
        >
          <Icon name="close" fill="#888" style={styles.closeIcon} />
        </TouchableOpacity>
        <Text category="s1" style={{ marginBottom: 8, marginTop: 8 }}>
          💡 What is Facial Age Estimation?
        </Text>
        <Text appearance="hint" style={{ marginBottom: 12 }}>
          This app uses deep learning models to estimate your biological age based on facial features.
        </Text>
        {modelDetails.map(model => (
          <View key={model.key} style={{ marginBottom: 18 }}>
            <Text style={{ fontWeight: 'bold', fontSize: 15 }}>{model.icon} {model.name}</Text>
            <Text style={{ fontSize: 13, color: '#444', marginTop: 4 }}>{model.summary}</Text>
            {expandedModel === model.key && (
              <View style={{ marginTop: 8, backgroundColor: '#f7f7f7', borderRadius: 8, padding: 10 }}>
                <Text style={{ fontSize: 12, color: '#333' }}>{model.details}</Text>
              </View>
            )}
            <TouchableOpacity onPress={() => setExpandedModel(expandedModel === model.key ? null : model.key)}>
              <Text style={{ color: '#2e7be4', marginTop: 6, fontSize: 12 }}>
                {expandedModel === model.key ? 'Show less' : 'Show more'}
              </Text>
            </TouchableOpacity>
          </View>
        ))}
      </Layout>
    );
    if (Platform.OS === 'web') {
      return infoVisible ? (
        <View style={styles.webModalOverlay}>
          {content}
        </View>
      ) : null;
    } else {
      return (
        <Modal
          visible={infoVisible}
          backdropStyle={{ backgroundColor: 'rgba(0,0,0,0.5)' }}
          onBackdropPress={() => setInfoVisible(false)}
        >
          {content}
        </Modal>
      );
    }
  };

  // Step 1: Upload or Take Photo
  const renderStep1 = () => (
    <Layout style={styles.stepContainer}>
      <Layout style={styles.contentContainer}>
        <View style={styles.mainContentContainer}>
          <Layout style={styles.headerContainer}>
            <Text category='h4' style={styles.stepTitle}>📸 Upload or Take A Photo</Text>
            <Text category='s1' style={styles.stepSubtitle}>
              Choose a clear photo with visible face(s)
            </Text>
          </Layout>

          <View style={styles.demoImageContainer}>
            <Image 
              source={require('./assets/demoimage.png')} 
              style={styles.demoImage}
              resizeMode="contain"
            />
          </View>

          <Layout style={styles.buttonContainer}>
            <Button
              style={styles.primaryButton}
              onPress={takePhoto}
              accessoryLeft={CameraIcon}
              size='large'
            >
              Take A Photo
            </Button>
            
            <Button
              style={styles.secondaryButton}
              onPress={pickImage}
              accessoryLeft={ImageIcon}
              size='large'
              status='basic'
            >
              Choose From Gallery
            </Button>
          </Layout>

          <Text style={{ textAlign: 'center', marginTop: 15, marginBottom: 10 }}>
            <TouchableOpacity onPress={() => setInfoVisible(true)} activeOpacity={0.7}>
              <Text style={{ color: '#3366FF' }}>
                💡 Learn how this works →
              </Text>
            </TouchableOpacity>
          </Text>
        </View>

        <Layout style={styles.apiStatusContainer}>
          {renderApiStatus()}
        </Layout>

        {renderInfoModal()}
      </Layout>
    </Layout>
  );

  // Step 2: Analyzing Photo
  const renderStep2 = () => (
    <Layout style={[styles.stepContainer, { minHeight: getResponsiveHeight(), maxHeight: getResponsiveHeight(), justifyContent: 'center' }]}> 
      <Layout style={styles.headerContainer}>
        <Text category='h4' style={styles.stepTitle}>🔍 Analyzing Photo</Text>
        <Text category='s1' style={styles.stepSubtitle}>
          Please wait while our models analyze your photo...
        </Text>
      </Layout>

      <Layout style={[styles.contentContainer, { minHeight: 0, maxHeight: getResponsiveHeight() * 0.7, justifyContent: 'center', alignItems: 'center' }]}> 
        {selectedImage && (
          <Layout style={[styles.analyzingImageContainer, { width: ANALYZE_IMAGE_SIZE, height: ANALYZE_IMAGE_SIZE }]}> 
            <Image 
              source={{ uri: selectedImage.uri }} 
              style={{
                width: ANALYZE_IMAGE_SIZE,
                height: ANALYZE_IMAGE_SIZE,
                borderRadius: 20,
              }}
              resizeMode="cover"
            />
            <View style={[styles.scanOverlay, { width: ANALYZE_IMAGE_SIZE, height: ANALYZE_IMAGE_SIZE }]}> 
              <Animated.View style={[
                {
                  position: 'absolute',
                  left: '50%',
                  width: ANALYZE_IMAGE_SIZE * 0.95,
                  height: 6,
                  borderRadius: 3,
                  shadowColor: '#4f8cff',
                  shadowOffset: { width: 0, height: 0 },
                  shadowOpacity: 0.5,
                  shadowRadius: 12,
                  elevation: 8,
                  opacity: scanLineOpacity,
                  transform: [
                    { translateX: -(ANALYZE_IMAGE_SIZE * 0.95) / 2 },
                    {
                      translateY: scanLinePosition.interpolate({
                        inputRange: [0, 1],
                        outputRange: [0, ANALYZE_IMAGE_SIZE - 6],
                      })
                    }
                  ]
                }
              ]}>
                <LinearGradient
                  colors={['#4f8cff', '#4fd1c5']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={{ width: '100%', height: '100%', borderRadius: 3 }}
                />
              </Animated.View>
              {/* Brackets inset to match border radius visually */}
              <View style={[styles.scanCornerTopLeft, { top: 20, left: 20 }]} />
              <View style={[styles.scanCornerTopRight, { top: 20, right: 20 }]} />
              <View style={[styles.scanCornerBottomLeft, { bottom: 20, left: 20 }]} />
              <View style={[styles.scanCornerBottomRight, { bottom: 20, right: 20 }]} />
            </View>
          </Layout>
        )}

        <Layout style={styles.loadingContainer}>
          <Spinner size='large' />
          <Text category='h6' style={styles.loadingText}>
            Detecting faces and analyzing age...
          </Text>
          <Text category='c1' style={styles.loadingSubtext}>
            Using Harvard FaceAge, DeepFace + OpenAI models
          </Text>
        </Layout>
      </Layout>
    </Layout>
  );

  // Step 3: Show Results
  const renderStep3 = () => (
    <Layout style={[styles.stepContainer, { paddingLeft: 0, paddingRight: 0, maxWidth: 500, width: '100%', alignSelf: 'center' }]}>
      <ScrollView style={[styles.resultsScrollView, { width: '100%', maxWidth: 500, alignSelf: 'center', paddingHorizontal: 0, marginHorizontal: 0 }]}>
        <Layout style={styles.headerContainer}>
          <Text category='h4' style={styles.stepTitle}>🎯 Analysis Results</Text>
          <Text category='s1' style={styles.stepSubtitle}>
            Age estimation complete
          </Text>
        </Layout>
        {results && results.faces && results.faces.filter(face => face.confidence >= 0.9).length > 0 ? (
          results.faces.filter(face => face.confidence >= 0.9).map((face, index) => {
            const filteredFaces = results.faces.filter(face => face.confidence >= 0.9);
            const isSingleFace = filteredFaces.length === 1;
            
            return (
              <Card key={index} style={[
                styles.resultCard,
                { width: '100%', alignSelf: 'center', marginLeft: 0, marginRight: 0 },
                isSingleFace && styles.singleFaceCard
              ]}>
                {!isSingleFace && (
                  <Layout style={styles.resultHeader}>
                    <Text category='h6' style={styles.resultTitle}>
                      Face {index + 1}
                    </Text>
                    <Text category='c1' style={styles.confidenceText}>
                      {(face.confidence * 100).toFixed(1)}% confidence
                    </Text>
                  </Layout>
                )}
                
                <Layout style={[styles.resultContent, isSingleFace && styles.singleFaceContent]}>
                {/* Face Analysis Grid */}
                <Layout style={styles.analysisGrid}>
                  {/* Face Detection Area */}
                  <Layout style={styles.faceDetectionArea}>
                    {!isSingleFace && (
                      <Text category='label' style={styles.sectionTitle}>FACE DETECTION</Text>
                    )}
                    {face.face_crop_base64 && (
                      <Layout style={styles.faceCropContainer}>
                        <View style={{
                          width: 200,
                          height: 200,
                          borderRadius: 20,
                          borderWidth: 10,
                          borderColor: '#fff',
                          shadowColor: '#000',
                          shadowOffset: { width: 0, height: 4 },
                          shadowOpacity: 0.1,
                          shadowRadius: 8,
                          elevation: 5,
                          backgroundColor: '#fff',
                          justifyContent: 'center',
                          alignItems: 'center',
                        }}>
                          <Image
                            source={{ uri: `data:image/jpeg;base64,${face.face_crop_base64}` }}
                            style={{ width: '100%', height: '100%', borderRadius: 20 }}
                            resizeMode="cover"
                          />
                        </View>
                        <View style={styles.faceOverlay}>
                          <View style={styles.detectionIndicator}>
                            <Text style={styles.detectionIcon}>✓</Text>
                          </View>
                        </View>
                      </Layout>
                    )}
                  </Layout>

                  {/* Age Estimation Results */}
                  <Layout style={{
                    marginTop: 14,
                    marginBottom: 14,
                    backgroundColor: 'white',
                    borderRadius: 12,
                    padding: 14,
                    borderWidth: 0,
                    shadowColor: '#000',
                    shadowOffset: { width: 0, height: 2 },
                    shadowOpacity: 0.04,
                    shadowRadius: 4,
                    elevation: 1,
                    width: '100%',
                    alignSelf: 'center',
                  }}>
                    <Text style={{ fontSize: 13, fontWeight: '700', color: '#4a5a6a', marginBottom: 8, letterSpacing: 0.2, textAlign: 'left' }}>AGE ESTIMATES</Text>
                    {/* Model rows */}
                    {(() => {
                      const modelRows = [];
                      if (face.age_harvard !== null && face.age_harvard !== undefined) modelRows.push({ key: 'harvard', value: face.age_harvard });
                      if (face.age_deepface !== null && face.age_deepface !== undefined) modelRows.push({ key: 'deepface', value: face.age_deepface });
                      if (face.age_chatgpt !== null && face.age_chatgpt !== undefined) modelRows.push({ key: 'chatgpt', value: face.age_chatgpt });
                      // Mean
                      const ages = modelRows.map(r => r.value);
                      const mean = ages.length ? (ages.reduce((a, b) => a + b, 0) / ages.length) : null;
                      return <>
                        {modelRows.map((row, i) => (
                          <Layout key={row.key} style={{ marginBottom: i === modelRows.length - 1 ? 0 : 12 }}>
                            <Layout style={{ flexDirection: 'row', alignItems: 'center', backgroundColor: 'transparent', minHeight: 36 }}>
                              <Text style={{ fontSize: 18, marginRight: 8 }}>{MODEL_ICONS[row.key]}</Text>
                              <Layout style={{ flex: 1 }}>
                                <Text style={{ fontWeight: '600', fontSize: 14, color: '#223', marginBottom: 1 }}>{MODEL_LABELS[row.key]}</Text>
                                <Text style={{ fontSize: 11, color: '#7a869a', marginBottom: 2 }}>{MODEL_DESCRIPTIONS[row.key]}</Text>
                              </Layout>
                              <Text style={{ fontWeight: 'bold', fontSize: 15 }}>{Math.round(row.value)}<Text style={{ fontSize: 12, color: '#888' }}> yrs</Text></Text>
                            </Layout>
                            <View style={{ position: 'relative', height: 7, width: '100%', borderRadius: 4, backgroundColor: '#e6eaf2', overflow: 'hidden', marginTop: 2, marginBottom: 2 }}>
                              {(() => {
                                const min = 0, max = 100;
                                const percent = Math.max(0, Math.min(1, ((row.value - min) / (max - min))));
                                return percent > 0 ? (
                                  <LinearGradient
                                    colors={['#4f8cff', '#4fd1c5']}
                                    start={{ x: 0, y: 0 }}
                                    end={{ x: 1, y: 0 }}
                                    style={{
                                      position: 'absolute',
                                      left: 0,
                                      top: 0,
                                      width: `${percent * 100}%`,
                                      height: '100%',
                                      borderRadius: 4,
                                      zIndex: 1,
                                    }}
                                  />
                                ) : null;
                              })()}
                            </View>
                            {i !== modelRows.length - 1 && (
                              <View style={{ height: 1, backgroundColor: '#eee', marginTop: 12, marginBottom: 12, marginLeft: 0 }} />
                            )}
                          </Layout>
                        ))}
                        {/* Separator and mean row */}
                        {mean !== null && (
                          <View style={{
                            marginTop: 18,
                            marginBottom: 4,
                            backgroundColor: '#e6f0fa',
                            borderRadius: 10,
                            padding: 12,
                            alignItems: 'center',
                            flexDirection: 'row',
                            justifyContent: 'center',
                            gap: 8,
                            shadowColor: '#4f8cff',
                            shadowOffset: { width: 0, height: 2 },
                            shadowOpacity: 0.08,
                            shadowRadius: 6,
                            elevation: 2,
                          }}>
                            <Text style={{ fontSize: 20, fontWeight: 'bold', color: '#1976d2', marginRight: 8 }}>📊</Text>
                            <Text style={{ fontSize: 16, fontWeight: '700', color: '#1976d2', marginRight: 8 }}>Mean Age</Text>
                            <Text style={{ fontSize: 22, fontWeight: 'bold', color: '#1976d2' }}>{Math.round(mean)}<Text style={{ fontSize: 13, color: '#888' }}> yrs</Text></Text>
                          </View>
                        )}
                      </>;
                    })()}
                  </Layout>
                </Layout>

                {/* Age Factors */}
                {face.chatgpt_factors && (
                  <Layout style={{
                    marginTop: 14,
                    marginBottom: 14,
                    backgroundColor: 'white',
                    borderRadius: 12,
                    padding: 14,
                    borderWidth: 0,
                    shadowColor: '#000',
                    shadowOffset: { width: 0, height: 2 },
                    shadowOpacity: 0.04,
                    shadowRadius: 4,
                    elevation: 1,
                    width: '100%',
                    alignSelf: 'center',
                  }}>
                    <Text style={{ fontSize: 13, fontWeight: '700', color: '#4a5a6a', marginBottom: 8, letterSpacing: 0.2, textAlign: 'left' }}>AGE FACTORS</Text>
                    {['skin_texture', 'skin_tone', 'hair', 'facial_volume'].map((factor, i, arr) => {
                      const f = face.chatgpt_factors[factor];
                      if (!f) return null;
                      const percent = Math.max(0, Math.min(1, ((f.age_rating - 0) / 100)));
                      return (
                        <Layout key={factor} style={{ marginBottom: i === arr.length - 1 ? 0 : 12 }}>
                          <Layout style={{ flexDirection: 'row', alignItems: 'center', backgroundColor: 'transparent', minHeight: 36 }}>
                            <Text style={{ fontSize: 20, marginRight: 8 }}>{AGE_FACTOR_ICONS[factor]}</Text>
                            <Layout style={{ flex: 1 }}>
                              <Text style={{ fontWeight: '600', fontSize: 14, color: '#223', marginBottom: 1 }}>{AGE_FACTOR_LABELS[factor]}</Text>
                              <Text style={{ fontSize: 11, color: '#7a869a', marginBottom: 2 }}>{f.explanation}</Text>
                            </Layout>
                            <Text style={{ fontWeight: 'bold', fontSize: 15, color: '#223', marginLeft: 10, minWidth: 32, textAlign: 'right' }}>{f.age_rating} <Text style={{ fontSize: 11, color: '#7a869a' }}>yrs</Text></Text>
                          </Layout>
                          <View style={{ position: 'relative', height: 7, width: '100%', borderRadius: 4, backgroundColor: '#e6eaf2', overflow: 'hidden', marginTop: 2, marginBottom: 2 }}>
                            <LinearGradient
                              colors={['#4f8cff', '#4fd1c5']}
                              start={{ x: 0, y: 0 }}
                              end={{ x: 1, y: 0 }}
                              style={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                width: `${percent * 100}%`,
                                height: '100%',
                                borderRadius: 4,
                                zIndex: 1,
                              }}
                            />
                          </View>
                          {i !== arr.length - 1 && (
                            <View style={{ height: 1, backgroundColor: '#eee', marginTop: 12, marginBottom: 12, marginLeft: 0 }} />
                          )}
                        </Layout>
                      );
                    })}
                  </Layout>
                )}
              </Layout>
            </Card>
          );
          })
        ) : (
          <Card style={styles.noResultsCard}>
            <Text category='h6' style={styles.noResultsText}>
              No clear faces detected
            </Text>
            <Text category='c1' style={styles.noResultsSubtext}>
              {results && results.faces && results.faces.length > 0 
                ? `Found ${results.faces.length} face(s) but none with sufficient confidence (≥90%)`
                : 'Try a clearer photo with visible faces'
              }
            </Text>
          </Card>
        )}
      </ScrollView>

      <Layout style={[styles.resultsActions, { maxWidth: 500, width: '100%', alignSelf: 'center' }]}>
        <Button
          style={styles.shareButton}
          onPress={shareResults}
          accessoryLeft={ShareIcon}
          status='primary'
        >
          Share Results
        </Button>
        
        <Button
          style={styles.secondaryButton}
          onPress={() => setCurrentStep(1)}
          accessoryLeft={ArrowBackIcon}
          status='basic'
        >
          Try Another Photo
        </Button>
      </Layout>
    </Layout>
  );

  const renderApiStatus = () => (
    <Layout style={{ alignItems: 'center', marginTop: 8 }}>
      <Text style={{ fontSize: 11, color: '#888', marginBottom: 6, fontWeight: '600', letterSpacing: 0.5 }}>Model Status</Text>
      <Layout style={{
        flexDirection: 'row',
        gap: 8,
        paddingVertical: 0,
        paddingHorizontal: 0,
        borderRadius: 0,
        backgroundColor: 'transparent',
        borderWidth: 0,
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: 20,
      }}>
        {['harvard', 'deepface', 'chatgpt'].map((model) => (
          <Layout key={model} style={{ flexDirection: 'row', alignItems: 'center', marginHorizontal: 2 }}>
            {apiHealth === null ? (
              <>
                <Spinner size='tiny' status='primary' style={{ width: 15, height: 15 }} />
                <Text style={{ fontSize: 12, fontWeight: '600', marginLeft: 3, color: '#888' }}>Loading...</Text>
              </>
            ) : (
              <>
                <Text style={{ fontSize: 15, color: apiHealth?.models?.[model] ? '#4CAF50' : '#f44336', fontWeight: 'bold' }}>
                  {apiHealth?.models?.[model] ? '✔' : '✖'}
                </Text>
                <Text style={{ fontSize: 12, fontWeight: '600', marginLeft: 3, color: '#222' }}>{MODEL_LABELS[model]}</Text>
              </>
            )}
          </Layout>
        ))}
      </Layout>
    </Layout>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="auto" backgroundColor="#ffffff" barStyle="dark-content" />
      <Layout style={styles.fullScreen}>
        {currentStep === 1 && renderStep1()}
        {currentStep === 2 && renderStep2()}
        {currentStep === 3 && renderStep3()}
      </Layout>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <Fragment>
      <IconRegistry icons={EvaIconsPack} />
      <ApplicationProvider {...eva} theme={eva.light}>
        <AppContent />
      </ApplicationProvider>
    </Fragment>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff', // pure white to match iPhone
  },
  fullScreen: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  stepContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: '#ffffff',
  },
  headerContainer: {
    alignItems: 'center',
    marginBottom: 15,
    paddingTop: 10,
  },
  stepTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a2a3a',
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 4,
  },
  stepSubtitle: {
    fontSize: 16,
    color: '#7a869a',
    textAlign: 'center',
    marginBottom: 12,
  },
  contentContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingTop: 20,
    paddingBottom: 20,
  },
  mainContentContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  apiStatusContainer: {
    alignItems: 'center',
    marginBottom: 10,
    marginTop: 5,
  },
  apiStatus: {
    fontSize: 14,
    fontWeight: 'normal',
  },
  apiConnected: {
    color: '#4CAF50',
  },
  apiDisconnected: {
    color: '#f44336',
  },
  demoImageContainer: {
    alignItems: 'center',
    marginBottom: 25,
  },
  demoImage: {
    width: Math.min(width * 0.85, 400),
    height: Math.min(width * 0.53, 400),
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  buttonContainer: {
    gap: 20,
    marginBottom: 20,
  },
  primaryButton: {
    borderRadius: 25,
    paddingVertical: 15,
  },
  secondaryButton: {
    backgroundColor: '#f6f8fa',
    borderRadius: 24,
    borderWidth: 0,
    marginBottom: 10,
    paddingVertical: 14,
  },
  analyzingImageContainer: {
    alignItems: 'center',
    marginBottom: 40,
    position: 'relative',
  },
  analyzingImage: {
    width: width * 0.7,
    height: width * 0.7,
    borderRadius: 20,
    overflow: 'hidden',
  },
  scanOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'flex-start',
    overflow: 'hidden',
  },
  scanLine: {
    height: 4,
    backgroundColor: '#3366FF',
    borderRadius: 2,
    shadowColor: '#3366FF',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 8,
    elevation: 5,
  },
  scanCornerTopLeft: {
    position: 'absolute',
    width: 18,
    height: 18,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#3366FF',
    borderTopLeftRadius: 8,
},
  scanCornerTopRight: {
    position: 'absolute',
    width: 18,
    height: 18,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderColor: '#3366FF',
    borderTopRightRadius: 8,
},
  scanCornerBottomLeft: {
    position: 'absolute',
    width: 18,
    height: 18,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#3366FF',
    borderBottomLeftRadius: 8,
},
  scanCornerBottomRight: {
    position: 'absolute',
    width: 18,
    height: 18,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderColor: '#3366FF',
    borderBottomRightRadius: 8,
},

  loadingContainer: {
    alignItems: 'center',
    gap: 15,
  },
  loadingText: {
    textAlign: 'center',
    marginTop: 10,
  },
  loadingSubtext: {
    textAlign: 'center',
    opacity: 0.6,
  },
  resultsScrollView: {
    flex: 1,
  },
  resultCard: {
    marginBottom: 15,
    borderRadius: 15,
  },
  singleFaceCard: {
    shadowOpacity: 0,
    elevation: 0,
    borderWidth: 0,
    backgroundColor: 'transparent',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  resultTitle: {
    fontWeight: 'bold',
  },
  confidenceText: {
    opacity: 0.7,
  },
  resultContent: {
    gap: 20,
  },
  analysisHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  analysisLabel: {
    color: '#3366FF',
    fontWeight: '600',
    letterSpacing: 1,
  },
  confidenceIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  confidenceBar: {
    height: 4,
    backgroundColor: '#3366FF',
    borderRadius: 2,
    minWidth: 20,
  },
  confidencePercentage: {
    color: '#3366FF',
    fontWeight: '600',
    fontSize: 12,
  },
  analysisGrid: {
    gap: 20,
  },
  faceDetectionArea: {
    gap: 12,
  },
  ageEstimationArea: {
    gap: 12,
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    letterSpacing: 1,
    marginBottom: 8,
  },
  faceCropContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 24,
    backgroundColor: 'transparent',
    position: 'relative',
  },
  faceCropImage: {
    width: 200,
    height: 200,
    borderRadius: 20,
    borderWidth: 5,
    borderColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    elevation: 12,
  },
  faceOverlay: {
    position: 'absolute',
    top: 20,
    left: 20,
    right: 20,
    bottom: 20,
    pointerEvents: 'none',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cornerTopLeft: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: 20,
    height: 20,
    borderTopWidth: 2,
    borderLeftWidth: 2,
    borderColor: '#3366FF',
  },
  cornerTopRight: {
    position: 'absolute',
    top: 0,
    right: 0,
    width: 20,
    height: 20,
    borderTopWidth: 2,
    borderRightWidth: 2,
    borderColor: '#3366FF',
  },
  cornerBottomLeft: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: 20,
    height: 20,
    borderBottomWidth: 2,
    borderLeftWidth: 2,
    borderColor: '#3366FF',
  },
  cornerBottomRight: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 20,
    height: 20,
    borderBottomWidth: 2,
    borderRightWidth: 2,
    borderColor: '#3366FF',
  },
  detectionIndicator: {
    backgroundColor: 'rgba(51, 102, 255, 0.9)',
    borderRadius: 20,
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#3366FF',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  detectionIcon: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  detectionStatus: {
    textAlign: 'center',
    color: '#3366FF',
    fontWeight: '500',
    fontSize: 12,
  },
  ageResultsContainer: {
    gap: 12,
  },
  ageResultCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e9ecef',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
    marginBottom: 14,
    width: '100%',
    alignSelf: 'center',
    paddingVertical: 14,
    paddingHorizontal: 16,
    minHeight: 56,
  },
  ageResultRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    paddingHorizontal: 4,
  },
  modelInfoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flexShrink: 1,
    minWidth: 0,
  },
  targetIcon: {
    fontSize: 22,
    marginRight: 2,
  },
  robotIcon: {
    fontSize: 22,
    marginRight: 2,
  },
  modelName: {
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 4,
    marginRight: 8,
    fontSize: 15,
    flexShrink: 1,
    minWidth: 0,
    maxWidth: 120,
    overflow: 'hidden',
  },
  modelSubtitle: {
    fontSize: 12,
    color: '#8a99b3', // light gray-blue
    marginTop: 2,
    marginBottom: 0,
    fontWeight: '400',
  },
  ageContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    minWidth: 80,
    justifyContent: 'flex-end',
  },
  infoIconContainer: {
    padding: 2,
    marginLeft: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  infoIcon: {
    width: 16,
    height: 16,
    color: '#3366FF', // consistent blue
  },
  tooltipText: {
    fontSize: 10,
    color: '#666',
    fontStyle: 'italic',
    marginTop: 4,
    marginBottom: 8,
    paddingHorizontal: 4,
  },
  modelBadge: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  badgeText: {
    fontSize: 9,
    fontWeight: '600',
    color: '#fff',
    letterSpacing: 0.5,
  },
  ageValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#5B9BD5', // lighter blue
    minWidth: 60,
    textAlign: 'right',
  },
  ageUnit: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 2,
  },
  analysisSummary: {
    gap: 12,
  },
  summaryTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    letterSpacing: 1,
  },
  summaryContent: {
    gap: 8,
  },
  summaryCard: {
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  summaryAge: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  summaryCategory: {
    fontSize: 11,
    color: '#3366FF',
    fontWeight: '500',
    letterSpacing: 0.5,
    marginBottom: 12,
  },
  ageIndicator: {
    width: '100%',
    height: 4,
    backgroundColor: '#e9ecef',
    borderRadius: 2,
    overflow: 'hidden',
  },
  ageBar: {
    height: '100%',
    backgroundColor: '#3366FF',
    borderRadius: 2,
  },
  noResultsCard: {
    alignItems: 'center',
    padding: 30,
    borderRadius: 15,
  },
  noResultsText: {
    textAlign: 'center',
    marginBottom: 10,
  },
  noResultsSubtext: {
    textAlign: 'center',
    opacity: 0.6,
  },
  resultsActions: {
    flexDirection: 'column',
    gap: 15,
    marginTop: 20,
    paddingHorizontal: 20, // add horizontal padding to match cards
  },
  shareButton: {
    backgroundColor: '#2196f3',
    borderRadius: 24,
    marginTop: 10,
    marginBottom: 8,
    paddingVertical: 14,
  },
  webCameraCard: {
    flex: 1,
    margin: 20,
  },
  howItWorksCard: {
    marginTop: 20,
    borderRadius: 16,
    padding: 20,
    backgroundColor: '#f8f9fa',
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  webCameraTitle: {
    textAlign: 'center',
    marginBottom: 20,
  },
  webCameraContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  video: {
    width: '100%',
    maxWidth: 400,
    height: 300,
    borderRadius: 10,
  },
  webCameraControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    gap: 10,
  },
  webCameraButton: {
    flex: 1,
  },
  infoModalCard: {
    borderRadius: 16,
    padding: 24,
    backgroundColor: '#f8f9fa',
    borderWidth: 1,
    borderColor: '#e9ecef',
    minWidth: 300,
    maxWidth: 340,
    alignSelf: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    zIndex: 1001,
  },
  webModalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  closeIconContainer: {
    position: 'absolute',
    top: 12,
    right: 12,
    zIndex: 1002,
    padding: 8,
  },
  closeIcon: {
    width: 32,
    height: 32,
  },
  summaryText: {
    fontSize: 16,
    color: '#555',
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 15,
  },
  confidenceValue: {
    fontSize: 12,
    color: '#3366FF',
    fontWeight: '600',
  },
});

