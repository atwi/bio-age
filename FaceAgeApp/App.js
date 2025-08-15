import React, { useState, useEffect, Fragment, useRef, Suspense, lazy } from 'react';
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
  Linking,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import * as Sharing from 'expo-sharing';
import { Ionicons } from '@expo/vector-icons';
import Constants from 'expo-constants';
import { LinearGradient } from 'expo-linear-gradient';
import { Asset } from 'expo-asset';

// Firebase imports
import { auth, db, storage, googleProvider, appleProvider } from './firebase';
import { signInWithGoogle, signInWithApple, signOut, createOrUpdateUserDocument, onAuthStateChange, getCurrentUser, sendMagicLink, isMagicLink, completeMagicLinkSignIn } from './services/authService';
import { getUserAnalysisHistory, getUserProfile, updateUserProfile, deleteAnalysis } from './services/userService';

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
  TopNavigation,
  TopNavigationAction,
} from '@ui-kitten/components';
import logo from './assets/logo.png';
import LoadingSpinner from './components/LoadingSpinner';
import GlassPanel from './components/GlassPanel';
import customDark from './theme/custom-dark.json';
import FAQAccordion from './components/FAQAccordion';

const { width, height } = Dimensions.get('window');
const MAIN_MAX_WIDTH = 500;

// Face outline dimensions with proper proportions
const FACE_WIDTH = width * 0.8;   // Larger so face fills most of the photo
const FACE_HEIGHT = FACE_WIDTH * 1.35; // Proper face proportions (1.35:1 ratio)

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
    console.log('🔍 Mobile API URL:', `http://${debuggerHost}:8001/api`);
    return `http://${debuggerHost}:8001/api`;
  } else {
    // For production mobile builds
    console.log('🔍 Production API URL:', 'https://trueage.app/api');
    return 'https://trueage.app/api';
  }
};

const API_BASE_URL = getApiBaseUrl();

// Helper to get base backend URL (without /api)
const getBackendBaseUrl = () => {
  const apiUrl = getApiBaseUrl();
  if (apiUrl.endsWith('/api')) {
    return apiUrl.slice(0, -4);
  }
  return apiUrl;
};

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

const ShieldAlertIcon = (props) => (
  <Icon {...props} name='alert-triangle-outline'/>
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
  harvard_calibrated: '🎯',
  deepface: '🤖',
  chatgpt: '💬',
  mean: '📊',
};
const MODEL_LABELS = {
  harvard: 'Harvard',
  harvard_calibrated: 'Harvard (calibrated)',
  deepface: 'DeepFace',
  chatgpt: 'GPT-Vision',
  mean: 'Mean Age',
};

const MODEL_DESCRIPTIONS = {
  harvard: 'Best for clinical/biological age estimation, people >40',
  harvard_calibrated: 'Harvard corrected via monotonic calibration (better under 40)',
  deepface: 'Best for general face age estimation',
  chatgpt: 'Best for human-like age perception',
};

const AppHeader = React.memo(function AppHeader({ onShowInfo, user, onSignIn, onSignOut }) {
  const openEmail = () => {
    const emailUrl = 'mailto:alexthorburnwinsor@gmail.com';
    if (Platform.OS === 'web') {
      window.open(emailUrl, '_self');
    } else {
      Linking.openURL(emailUrl);
    }
  };

  const InfoAction = () => (
    <TopNavigationAction
      icon={(props) => <Icon {...props} name='info-outline' />}
      onPress={onShowInfo}
    />
  );

  
  return (
    <TopNavigation
      alignment='center'
      title={() => (
        <TouchableOpacity style={styles.headerRow} onPress={() => {
          if (Platform.OS === 'web') {
            window.location.href = '/';
          }
        }}>
          <Image 
            source={logo} 
            style={styles.headerLogo} 
            defaultSource={Platform.OS === 'ios' ? logo : undefined} // fallback for iOS
          />
          <Text category='h6' style={styles.headerTitle}>TrueAge</Text>
        </TouchableOpacity>
      )}
      accessoryRight={() => (
        <Layout style={{ flexDirection: 'row', alignItems: 'center' }}>
          {user ? (
            <>
              <Text style={{ fontSize: 12, color: '#666', marginRight: 8 }}>
                {user.displayName}
              </Text>
              <TopNavigationAction
                icon={(props) => <Icon {...props} name='log-out-outline' />}
                onPress={onSignOut}
              />
            </>
          ) : (
            <>
              <TopNavigationAction
                icon={(props) => <Icon {...props} name='person-outline' />}
                onPress={() => onSignIn('google')}
              />
            </>
          )}
          <InfoAction />
        </Layout>
      )}
      style={styles.headerNav}
    />
  );
});

const AppFooter = ({ onShowModal, onShowInfo }) => {
  const showModal = (type) => {
    onShowModal(type);
  };

  const openEmail = () => {
    const emailUrl = 'mailto:alexthorburnwinsor@gmail.com';
    if (Platform.OS === 'web') {
      window.open(emailUrl, '_self');
    } else {
      Linking.openURL(emailUrl);
    }
  };

  return (
    <Layout style={styles.footer}>
      <Layout style={styles.footerContent}>
        <Layout style={styles.footerLinks}>
          <TouchableOpacity onPress={onShowInfo}>
            <Text style={styles.footerLink}>How It Works</Text>
          </TouchableOpacity>
          <Text style={styles.footerSeparator}>•</Text>
          <TouchableOpacity onPress={() => showModal('privacy')}>
            <Text style={styles.footerLink}>Privacy Policy</Text>
          </TouchableOpacity>
          <Text style={styles.footerSeparator}>•</Text>
          <TouchableOpacity onPress={() => showModal('terms')}>
            <Text style={styles.footerLink}>Terms of Service</Text>
          </TouchableOpacity>
          <Text style={styles.footerSeparator}>•</Text>
          <TouchableOpacity onPress={openEmail}>
            <Text style={styles.footerLink}>Contact</Text>
          </TouchableOpacity>
        </Layout>
        <Text style={styles.footerCopyright}>
          © 2025 TrueAge. Built with Harvard FaceAge, DeepFace & OpenAI.
        </Text>
      </Layout>
    </Layout>
  );
};

function AppContent() {
  const [currentStep, setCurrentStep] = useState(1); // 1: Upload, 2: Analyzing, 3: Results
  const [selectedImage, setSelectedImage] = useState(null);
  const [sourceIsSelfie, setSourceIsSelfie] = useState(false); // true if captured in-app (selfie), false if uploaded
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [showWebCamera, setShowWebCamera] = useState(false);
  const [webCameraStream, setWebCameraStream] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const liveOverlayRef = useRef(null);
  const dotAlphaRef = useRef(0); // for smooth fade-in/out of landmark dots
  const lastAlignedRef = useRef(false); // track alignment lock transitions
  const pulseUntilRef = useRef(0); // time until which to show the brief glow pulse
  const colorMixRef = useRef(0); // 0 = brand gradient, 1 = green; eased per frame for smooth color fade
  const badgeScale = useRef(new Animated.Value(0.96)).current; // instruction badge scale
  const badgeOpacity = useRef(new Animated.Value(0.9)).current; // instruction badge opacity
  const [guidanceMessage, setGuidanceMessage] = useState('Align your face in the frame');
  const [isAligned, setIsAligned] = useState(false);
  
  // Animated values for scanning effect
  const scanLinePosition = useRef(new Animated.Value(0)).current;
  const scanLineOpacity = useRef(new Animated.Value(0.3)).current;
  const [infoVisible, setInfoVisible] = useState(false);
  const [harvardTooltipVisible, setHarvardTooltipVisible] = useState(false);
  const [deepfaceTooltipVisible, setDeepfaceTooltipVisible] = useState(false);
  const [chatgptTooltipVisible, setChatgptTooltipVisible] = useState(false);
  const [showApiTooltip, setShowApiTooltip] = useState(false);
  const [expandedModel, setExpandedModel] = useState(null);
  const [faceMeshOverlays, setFaceMeshOverlays] = useState({});

  const [modalContent, setModalContent] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  
  // Firebase auth state
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const emailInputRef = useRef(null);
  const [emailForLink, setEmailForLink] = useState('');
  const [emailSending, setEmailSending] = useState(false);
  const [emailSent, setEmailSent] = useState(false);

  // Live AR overlay state (web)
  const [livePrediction, setLivePrediction] = useState(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [facePosition, setFacePosition] = useState(null); // { x, y, width, height }
  const [analysisProgress, setAnalysisProgress] = useState(0); // 0-100 for loading ring
  const [showFullResultsButton, setShowFullResultsButton] = useState(false);
  const [hasCompletedAnalysis, setHasCompletedAnalysis] = useState(false); // Track if analysis is complete
  const [arAnalysisResults, setArAnalysisResults] = useState(null); // Store AR analysis results
  const [arCapturedImage, setArCapturedImage] = useState(null); // Store captured image from AR
  const lastFaceBoxRef = useRef(null); // {minX, minY, maxX, maxY}
  const analyzingLiveRef = useRef(false);
  const lastAnalyzeTimeRef = useRef(0);
  const autoAnalysisTimeoutRef = useRef(null);
  const wasAlignedRef = useRef(false); // Track previous alignment state

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

  // Set up Firebase auth state listener
  useEffect(() => {
    const unsubscribe = onAuthStateChange((user) => {
      setUser(user);
      setAuthLoading(false);
    });

    return () => unsubscribe();
  }, []);

  // Complete email link sign-in if opened via magic link (web)
  useEffect(() => {
    if (Platform.OS === 'web') {
      try {
        const href = window.location.href;
        if (isMagicLink(href)) {
          // Attempt to complete with stored email; UI will ask if missing
          completeMagicLinkSignIn(href)
            .then(() => {
              // Optionally clean up URL params
              if (window.history && window.location) {
                const url = new URL(window.location.href);
                url.search = '';
                window.history.replaceState({}, document.title, url.toString());
              }
            })
            .catch(() => {});
        }
      } catch (e) {}
    }
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
        
        // Console log model status instead of displaying in UI
        console.log('🔍 Model Status:', {
          harvard: data.models?.harvard ? '✅ Ready' : '❌ Not Ready',
          harvard_calibrated: data.models?.harvard_calibrated ? '✅ Ready' : '❌ Not Ready',
          deepface: data.models?.deepface ? '✅ Ready' : '❌ Not Ready', 
          chatgpt: data.models?.chatgpt ? '✅ Ready' : '❌ Not Ready',
          loading: data.models_loading ? '🔄 Loading...' : '✅ Complete',
          ready_for_analysis: data.ready_for_analysis ? '✅ Ready' : '❌ Not Ready'
        });
        
        // If models are still loading, check again in 2 seconds
        if (data.models_loading) {
          setTimeout(() => checkApiHealth(), 2000);
        }
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
        setSourceIsSelfie(false);
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
          setSourceIsSelfie(true);
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
        // Add event listener to ensure video is ready
        videoRef.current.addEventListener('loadedmetadata', () => {
          console.log('[LiveAR] Video metadata loaded:', {
            width: videoRef.current.videoWidth,
            height: videoRef.current.videoHeight,
            readyState: videoRef.current.readyState
          });
        });
        videoRef.current.addEventListener('canplay', () => {
          console.log('[LiveAR] Video can play');
        });
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
          setSourceIsSelfie(true);
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
    // Clear any pending auto-analysis timeout
    if (autoAnalysisTimeoutRef.current) {
      console.log('[LiveAR] Clearing timeout - camera closed');
      clearTimeout(autoAnalysisTimeoutRef.current);
      autoAnalysisTimeoutRef.current = null;
    }
    // Clear stored AR results when camera is closed
    setArAnalysisResults(null);
    setArCapturedImage(null);
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

  // Live web: capture the current frame and analyze without changing steps
  const analyzeFaceLive = async (imageObj) => {
    let progressInterval = null;
    try {
      console.log('[LiveAR] Starting analyzeFaceLive with:', imageObj);
      setLiveLoading(true);
      setAnalysisProgress(0);
      
      // Simulate progress updates with smoother animation
      progressInterval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 90) return prev; // Stop at 90% until API responds
          // Smoother progress with smaller, more frequent increments
          return Math.min(90, prev + (Math.random() * 3 + 1)); // 1-4% increments
        });
      }, 100); // More frequent updates for smoother animation
      
      const formData = new FormData();
      if (typeof imageObj?.uri === 'string' && (imageObj.uri.startsWith('data:') || imageObj.uri.startsWith('blob:'))) {
        console.log('[LiveAR] Processing blob/data URL');
        const resp = await fetch(imageObj.uri);
        const blob = await resp.blob();
        console.log('[LiveAR] Fetched blob, size:', blob.size);
        formData.append('file', blob, 'frame.jpg');
      } else if (imageObj?.uri) {
        console.log('[LiveAR] Processing URI');
        formData.append('file', { uri: imageObj.uri, type: 'image/jpeg', name: 'frame.jpg' });
      } else if (imageObj?.blob) {
        console.log('[LiveAR] Processing direct blob');
        formData.append('file', imageObj.blob, 'frame.jpg');
      } else {
        console.log('[LiveAR] No valid image object provided');
        setLiveLoading(false);
        analyzingLiveRef.current = false;
        if (progressInterval) clearInterval(progressInterval);
        return;
      }

      console.log('[LiveAR] Sending request to API...');
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);
      const response = await fetch(`${API_BASE_URL}/analyze-face`, { method: 'POST', body: formData, signal: controller.signal });
      clearTimeout(timeoutId);

      console.log('[LiveAR] API response status:', response.status);
      if (response.ok) {
        const data = await response.json();
        console.log('[LiveAR] API response data:', data);
        let ageToShow = null;
        if (data && data.faces && data.faces.length > 0) {
          const f = data.faces[0];
          if (f.age_harvard_calibrated != null) {
            ageToShow = Number(f.age_harvard_calibrated);
          } else {
            const ages = [];
            if (f.age_harvard != null) ages.push(Number(f.age_harvard));
            if (f.age_deepface != null) ages.push(Number(f.age_deepface));
            if (f.age_chatgpt != null) ages.push(Number(f.age_chatgpt));
            if (ages.length > 0) ageToShow = ages.reduce((a, b) => a + b, 0) / ages.length;
          }
        }
        console.log('[LiveAR] Calculated age to show:', ageToShow);
        
        // Clear progress interval before completing
        if (progressInterval) clearInterval(progressInterval);
        
        // Complete the progress animation
        setAnalysisProgress(100);
        setTimeout(() => {
          setLivePrediction(ageToShow != null && !Number.isNaN(ageToShow) ? ageToShow : null);
          setShowFullResultsButton(true);
          setHasCompletedAnalysis(true); // Mark analysis as completed
          // Store the full analysis results for reuse
          setArAnalysisResults(data);
        }, 300); // Small delay for smooth transition
        
      } else {
        console.log('[LiveAR] API request failed with status:', response.status);
        const errorText = await response.text();
        console.log('[LiveAR] API error response:', errorText);
        setLivePrediction(null);
      }
    } catch (e) {
      console.log('[LiveAR] Error in analyzeFaceLive:', e);
      setLivePrediction(null);
    } finally {
      console.log('[LiveAR] Analysis completed, setting loading to false');
      setLiveLoading(false);
      analyzingLiveRef.current = false;
      // Always clear the progress interval
      if (progressInterval) clearInterval(progressInterval);
    }
  };

  const triggerLiveAnalysisFromVideo = (videoEl) => {
    try {
      const canvas = canvasRef.current;
      if (!videoEl || !canvas) {
        console.log('[LiveAR] Missing video element or canvas:', { videoEl: !!videoEl, canvas: !!canvas });
        return;
      }
      console.log('[LiveAR] Capturing frame for analysis');
      console.log('[LiveAR] Video dimensions:', { width: videoEl.videoWidth, height: videoEl.videoHeight });
      
      if (!videoEl.videoWidth || !videoEl.videoHeight) {
        console.log('[LiveAR] Video not ready - no dimensions');
        analyzingLiveRef.current = false;
        return;
      }
      
      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      const ctx2 = canvas.getContext('2d');
      ctx2.drawImage(videoEl, 0, 0);
      
      console.log('[LiveAR] Canvas created, creating blob...');
      canvas.toBlob((blob) => {
        if (!blob) { 
          console.log('[LiveAR] Failed to create blob');
          analyzingLiveRef.current = false; 
          return; 
        }
        console.log('[LiveAR] Blob created, size:', blob.size);
        const url = URL.createObjectURL(blob);
        console.log('[LiveAR] Starting analysis with URL:', url);
        analyzeFaceLive({ uri: url }).finally(() => { 
          try { 
            URL.revokeObjectURL(url); 
            console.log('[LiveAR] Analysis completed, URL revoked');
          } catch (_) {} 
        });
      }, 'image/jpeg', 0.85);
    } catch (error) {
      console.log('[LiveAR] Error in triggerLiveAnalysisFromVideo:', error);
      analyzingLiveRef.current = false;
    }
  };

  // Lazy-load FaceMesh and render cyan dots only when web camera is open
  useEffect(() => {
    if (!(showWebCamera && Platform.OS === 'web')) return;
    let faceMeshInstance = null;
    let rafId = 0;
    let running = true;

    // Compute convex hull (monotonic chain) to avoid interior crossing edges
    const hull = (pts) => {
      if (!pts || pts.length < 3) return pts || [];
      const points = pts.slice().sort((a, b) => (a.x - b.x) || (a.y - b.y));
      const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
      const lower = [];
      for (const p of points) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
        lower.push(p);
      }
      const upper = [];
      for (let i = points.length - 1; i >= 0; i--) {
        const p = points[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
        upper.push(p);
      }
      upper.pop();
      lower.pop();
      return lower.concat(upper);
    };

    // Chaikin corner-cutting to smoothly refine a closed polyline
    const smoothClosedPolyline = (pts, iterations = 3) => {
      if (!pts || pts.length < 3) return pts || [];
      let poly = pts.slice();
      for (let it = 0; it < iterations; it++) {
        const next = [];
        const n = poly.length;
        for (let i = 0; i < n; i++) {
          const a = poly[i];
          const b = poly[(i + 1) % n];
          // Q and R points
          next.push({ x: 0.75 * a.x + 0.25 * b.x, y: 0.75 * a.y + 0.25 * b.y });
          next.push({ x: 0.25 * a.x + 0.75 * b.x, y: 0.25 * a.y + 0.75 * b.y });
        }
        poly = next;
      }
      return poly;
    };

    const init = async () => {
      try {
        const video = videoRef.current;
        if (!video) return;
        const { FaceMesh } = await import('@mediapipe/face_mesh');
        faceMeshInstance = new FaceMesh({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        faceMeshInstance.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        faceMeshInstance.onResults((res) => {
          const canvas = liveOverlayRef.current;
          if (!canvas || !video) return;
          const w = video.videoWidth || 0;
          const h = video.videoHeight || 0;
          if (!w || !h) return;
          const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1;
          if (canvas.width !== Math.floor(w * dpr)) canvas.width = Math.floor(w * dpr);
          if (canvas.height !== Math.floor(h * dpr)) canvas.height = Math.floor(h * dpr);
          const ctx = canvas.getContext('2d');
          // Normalize drawing coordinates to CSS pixels
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, w, h);
          const faces = res.multiFaceLandmarks || [];
          if (!faces.length) {
            setGuidanceMessage('Align your face in the frame');
            setIsAligned(false);
            // Don't reset wasAlignedRef here - only reset it when analysis is complete
            setFacePosition(null); // Clear face position when no face detected
            
            // NEVER clear analysis results when face is lost - only clear on Try Again or page reload
            // The results should persist even when face moves out of frame
            
            // Clear any pending auto-analysis timeout
            if (autoAnalysisTimeoutRef.current) {
              console.log('[LiveAR] Clearing timeout - face lost');
              clearTimeout(autoAnalysisTimeoutRef.current);
              autoAnalysisTimeoutRef.current = null;
            }
            // Smoothly fade dots out when face is lost
            dotAlphaRef.current += (0 - dotAlphaRef.current) * 0.2;
            return;
          }
          // Adaptive guidance: size/position feedback
          let message = 'Align your face in the frame';
          try {
            const pts = faces[0].map(({ x, y }) => ({ x: x * w, y: y * h }));
            if (pts.length > 0) {
              let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
              for (const p of pts) {
                if (p.x < minX) minX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.x > maxX) maxX = p.x;
                if (p.y > maxY) maxY = p.y;
              }
              const faceW = Math.max(1, maxX - minX);
              const faceH = Math.max(1, maxY - minY);
              const faceAreaFrac = (faceW * faceH) / (w * h);
              const cx = (minX + maxX) / 2;
              const cy = (minY + maxY) / 2;
              const cxFrac = Math.abs(cx - w / 2) / w;
              const cyFrac = Math.abs(cy - h / 2) / h;
              
              // Update face position for AR display
              setFacePosition({
                x: cx,
                y: cy,
                width: faceW,
                height: faceH,
                minX,
                minY,
                maxX,
                maxY
              });
              
              if (faceAreaFrac < 0.12) {
                message = 'Move closer';
              } else if (faceAreaFrac > 0.50) {
                message = 'Move farther';
              } else if (cxFrac > 0.20 || cyFrac > 0.20) {
                message = 'Center your face';
              } else {
                message = 'Ready to capture';
              }
              const isCurrentlyAligned = message === 'Ready to capture';
              const wasAligned = wasAlignedRef.current;
              setIsAligned(isCurrentlyAligned);
              wasAlignedRef.current = isCurrentlyAligned;
              
                            // Auto-trigger analysis when face becomes aligned and not already analyzing
              if (isCurrentlyAligned && !wasAligned && !analyzingLiveRef.current && !liveLoading && !hasCompletedAnalysis) {
                console.log('[LiveAR] Face aligned, preparing to auto-trigger analysis');
                // Small delay to ensure face is stable
                if (autoAnalysisTimeoutRef.current) {
                  console.log('[LiveAR] Clearing existing timeout');
                  clearTimeout(autoAnalysisTimeoutRef.current);
                }
                console.log('[LiveAR] Setting new timeout');
                autoAnalysisTimeoutRef.current = setTimeout(() => {
                  console.log('[LiveAR] Auto-trigger timeout fired!');
                  const v = videoRef.current;
                  console.log('[LiveAR] Auto-trigger timeout fired, checking conditions:', {
                    hasVideo: !!v,
                    analyzingLive: analyzingLiveRef.current,
                    liveLoading,
                    videoReady: v ? (v.videoWidth && v.videoHeight && v.readyState >= 2) : false
                  });
                  if (v && !analyzingLiveRef.current && !liveLoading && v.videoWidth && v.videoHeight && v.readyState >= 2) {
                    console.log('[LiveAR] Auto-triggering analysis');
                    analyzingLiveRef.current = true;
                    lastAnalyzeTimeRef.current = Date.now();
                    setLivePrediction(null);
                    setLiveLoading(true);
                    setAnalysisProgress(0);
                    setShowFullResultsButton(false);
                    triggerLiveAnalysisFromVideo(v);
                  } else {
                    console.log('[LiveAR] Auto-trigger conditions not met, not starting analysis');
                  }
                }, 1000); // 1 second delay to ensure face is stable
                console.log('[LiveAR] Timeout set, waiting 1 second...');
              } else if (isCurrentlyAligned && wasAligned) {
                console.log('[LiveAR] Face still aligned, no need to trigger again');
              } else if (!isCurrentlyAligned) {
                console.log('[LiveAR] Face not ready:', message);
              }
            }
          } catch (_) {}

          // Update UI message (rendered in the top overlay, not on canvas)
          setGuidanceMessage(message);

          // No face outline drawing (badge provides alignment feedback)
          try { /* intentionally empty */ } catch (_) {}
        });

        const loop = async () => {
          if (!running) return;
          if (video.readyState >= 2) {
            await faceMeshInstance.send({ image: video });
          }
          rafId = requestAnimationFrame(loop);
        };
        rafId = requestAnimationFrame(loop);
      } catch (e) {
        console.error('FaceMesh init error:', e);
      }
    };

    init();
    return () => {
      running = false;
      if (rafId) cancelAnimationFrame(rafId);
      try { faceMeshInstance && faceMeshInstance.close && faceMeshInstance.close(); } catch {}
      const c = liveOverlayRef.current;
      if (c) {
        const ctx = c.getContext('2d');
        ctx && ctx.clearRect(0, 0, c.width || 0, c.height || 0);
      }
    };
  }, [showWebCamera, apiHealth?.settings?.enable_face_fill]);

  const analyzeFace = async (imageToAnalyze = selectedImage, { overrideLowConfidence = false } = {}) => {
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

      console.log('Sending request to:', `${API_BASE_URL}/analyze-face${overrideLowConfidence ? '?allow_low_confidence=true' : ''}`);

      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout
      
      const response = await fetch(`${API_BASE_URL}/analyze-face${overrideLowConfidence ? '?allow_low_confidence=true' : ''}`, {
        method: 'POST',
        body: formData,
        headers: overrideLowConfidence ? { 'x-allow-low-confidence': 'true' } : undefined,
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

  const handleSignIn = async (provider) => {
    try {
      let result;
      if (provider === 'google') {
        result = await signInWithGoogle();
      } else if (provider === 'apple') {
        result = await signInWithApple();
      }
      
      if (result) {
        await createOrUpdateUserDocument(result.user);
        Alert.alert('Success', 'Signed in successfully!');
      }
    } catch (error) {
      console.error('Sign in error:', error);
      Alert.alert('Error', 'Failed to sign in: ' + error.message);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      Alert.alert('Success', 'Signed out successfully!');
    } catch (error) {
      console.error('Sign out error:', error);
      Alert.alert('Error', 'Failed to sign out: ' + error.message);
    }
  };

  const [toast, setToast] = useState({ visible: false, text: '' });
  const toastOpacity = useRef(new Animated.Value(0)).current;
  const showToast = (text) => {
    setToast({ visible: true, text });
    // Fade in
    Animated.timing(toastOpacity, { toValue: 1, duration: 180, useNativeDriver: true }).start(() => {
      // Hold, then fade out
      setTimeout(() => {
        Animated.timing(toastOpacity, { toValue: 0, duration: 220, useNativeDriver: true }).start(() => {
          setToast({ visible: false, text: '' });
        });
      }, 1200);
    });
  };

  const shareResults = async () => {
    if (!results || !results.faces || results.faces.length === 0) {
      Alert.alert('Error', 'No results to share');
      return;
    }

    const shareText = generateShareText();
    const shareUrl = 'https://trueage.app';

    try {
      if (Platform.OS === 'ios' || Platform.OS === 'android') {
        // Use React Native's Share API for mobile platforms
        await Share.share({
          message: shareText,
          title: 'My Age Analysis Results',
          url: shareUrl,
        });
      } else if (Platform.OS === 'web' && typeof navigator !== 'undefined' && navigator.share) {
        // Web Share API (Chrome/Edge mobile/desktop)
        await navigator.share({ title: 'My Age Analysis Results', text: shareText, url: shareUrl });
      } else if (Platform.OS === 'web') {
        // Firefox/Safari desktop fallback → copy to clipboard and show toast
        await copyToClipboard(shareText);
        showToast('Results copied to clipboard');
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
        if (Platform.OS === 'web') {
          showToast('Results copied to clipboard');
        } else {
          Alert.alert('Copied!', 'Results copied to clipboard since sharing failed.');
        }
      } catch (clipboardError) {
        Alert.alert('Share Results', shareText, [
          { text: 'OK', style: 'default' }
        ]);
      }
    }
  };

  const generateShareText = () => {
    if (!results || !results.faces) return '';
    
    const validFaces = results.override_used ? results.faces : results.faces.filter(face => face.confidence >= 0.9);
    if (validFaces.length === 0) return results && results.message ? results.message : 'No clear faces detected in the analysis.';

    let shareText = '🎯 My Age Analysis Results:\n\n';
    
    validFaces.forEach((face, index) => {
      shareText += `Face ${index + 1}:\n`;
      // Compute Consensus as simple average of all shown models
      const modelAges = [];
      if (face.age_harvard !== null && face.age_harvard !== undefined) modelAges.push(Number(face.age_harvard));
      if (face.age_harvard_calibrated !== null && face.age_harvard_calibrated !== undefined) modelAges.push(Number(face.age_harvard_calibrated));
      if (face.age_deepface !== null && face.age_deepface !== undefined) modelAges.push(Number(face.age_deepface));
      if (face.age_chatgpt !== null && face.age_chatgpt !== undefined) modelAges.push(Number(face.age_chatgpt));
      if (modelAges.length > 0) {
        const consensus = modelAges.reduce((a, b) => a + b, 0) / modelAges.length;
        shareText += `⭐ Consensus: ${consensus.toFixed(1)} years\n`;
      }
      if (face.age_harvard !== null && face.age_harvard !== undefined) shareText += `Harvard: ${Number(face.age_harvard).toFixed(1)} years\n`;
      if (face.age_harvard_calibrated !== null && face.age_harvard_calibrated !== undefined) shareText += `Harvard (calibrated): ${Number(face.age_harvard_calibrated).toFixed(1)} years\n`;
      if (face.age_deepface !== null && face.age_deepface !== undefined) shareText += `DeepFace: ${Number(face.age_deepface).toFixed(1)} years\n`;
      if (face.age_chatgpt !== null && face.age_chatgpt !== undefined) shareText += `ChatGPT: ${Number(face.age_chatgpt).toFixed(1)} years\n`;
      if (face.confidence !== null && face.confidence !== undefined) shareText += `Confidence: ${(Number(face.confidence) * 100).toFixed(1)}%\n`;
      shareText += `\n`;
    });

    shareText += `Try TrueAge: https://trueage.app`;
    return shareText;
  };

  const copyToClipboard = async (text) => {
    try {
      if (Platform.OS === 'web') {
        const canUseAsyncClipboard = typeof navigator !== 'undefined' && navigator.clipboard && (window.isSecureContext || location.hostname === 'localhost');
        if (canUseAsyncClipboard) {
          await navigator.clipboard.writeText(text);
        } else {
          // Fallback: hidden textarea + execCommand('copy') works in Firefox/older browsers
          const textarea = document.createElement('textarea');
          textarea.value = text;
          textarea.setAttribute('readonly', '');
          textarea.style.position = 'fixed';
          textarea.style.top = '-1000px';
          textarea.style.opacity = '0';
          document.body.appendChild(textarea);
          textarea.focus();
          textarea.select();
          const ok = document.execCommand && document.execCommand('copy');
          document.body.removeChild(textarea);
          if (!ok) {
            throw new Error('Fallback execCommand copy failed');
          }
        }
      } else {
        // For mobile, we could use expo-clipboard if needed
        console.log('Clipboard text:', text);
      }
    } catch (error) {
      console.error('Clipboard error:', error);
      throw error;
    }
  };

  const getResponsiveHeight = () => {
    if (typeof window !== 'undefined' && window.innerHeight) {
      return window.innerHeight;
    }
    return height;
  };

  const fetchFaceMeshOverlay = async (faceCropBase64, faceIndex) => {
    try {
      // Convert base64 to blob
      const byteCharacters = atob(faceCropBase64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/jpeg' });
      const formData = new FormData();
      formData.append('file', blob, 'face.jpg');
      
      // Always use the combined visualization endpoint
      const response = await fetch(`${API_BASE_URL.replace('/api', '')}/facemesh-overlay`, {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        setFaceMeshOverlays(prev => ({ ...prev, [faceIndex]: data.image_base64 }));
      } else {
        setFaceMeshOverlays(prev => ({ ...prev, [faceIndex]: null }));
      }
    } catch (e) {
      setFaceMeshOverlays(prev => ({ ...prev, [faceIndex]: null }));
    }
  };

  useEffect(() => {
    if (results && results.faces) {
      results.faces.forEach((face, i) => {
        if (face.face_crop_base64) {
          fetchFaceMeshOverlay(face.face_crop_base64, i);
        }
      });
    }
  }, [results]);

  // Animate the glassy instruction badge for clearer alignment feedback
  useEffect(() => {
    if (isAligned) {
      Animated.parallel([
        Animated.timing(badgeOpacity, { toValue: 1, duration: 180, useNativeDriver: false }),
        Animated.sequence([
          Animated.timing(badgeScale, { toValue: 1.06, duration: 120, useNativeDriver: false }),
          Animated.timing(badgeScale, { toValue: 1.0, duration: 120, useNativeDriver: false }),
        ]),
      ]).start();
    } else {
      Animated.parallel([
        Animated.timing(badgeOpacity, { toValue: 0.9, duration: 150, useNativeDriver: false }),
        Animated.timing(badgeScale, { toValue: 0.96, duration: 150, useNativeDriver: false }),
      ]).start();
    }
  }, [isAligned]);

  // Web camera view for browsers
  if (showWebCamera && Platform.OS === 'web') {
    return (
      <View style={styles.fullScreenCamera}>
        {/* CSS for loading animation */}
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
        <video
          ref={videoRef}
          style={styles.fullScreenVideo}
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={liveOverlayRef}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', transform: 'scaleX(-1)', zIndex: 4 }}
        />
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
        
        {/* Top overlay with title */}
        <View style={styles.cameraTopOverlay}>
          {livePrediction === null && (
            <Animated.View
              style={[
                styles.instructionContainer,
                {
                  opacity: badgeOpacity,
                  transform: [{ scale: badgeScale }],
                  backdropFilter: 'blur(10px)',
                  backgroundColor: liveLoading ? 'rgba(0, 220, 140, 0.18)' : (isAligned ? 'rgba(0, 220, 140, 0.18)' : 'rgba(0, 0, 0, 0.28)'),
                  borderWidth: 1,
                  borderColor: liveLoading ? 'rgba(0, 220, 140, 0.45)' : (isAligned ? 'rgba(0, 220, 140, 0.45)' : 'rgba(255, 255, 255, 0.28)'),
                  filter: (liveLoading || isAligned) ? 'drop-shadow(0 0 10px rgba(0,220,140,0.35))' : 'none',
                },
              ]}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                {liveLoading ? (
                  <Ionicons name="scan-outline" size={16} color="rgba(0,220,140,0.95)" />
                ) : isAligned ? (
                  <Ionicons name="checkmark-circle" size={16} color="rgba(0,220,140,0.95)" />
                ) : (
                  <Ionicons name="scan-outline" size={16} color="rgba(255,255,255,0.9)" />
                )}
                <Text style={styles.instructionText}>
                  {liveLoading ? 'Scanning face...' : guidanceMessage}
                </Text>
              </View>
            </Animated.View>
          )}
        </View>
        
        {/* Face Outline Overlay */}
        <View style={styles.faceOutlineContainer} />
        
        {/* Loading Ring Around Face */}
        {liveLoading && facePosition && videoRef.current && (
          <View style={{
            position: 'absolute',
            left: (videoRef.current.videoWidth || 640) - facePosition.x - facePosition.width/2 - 20,
            top: facePosition.minY - 20,
            width: facePosition.width + 40,
            height: facePosition.height + 40,
            zIndex: 5,
            pointerEvents: 'none'
          }}>
            {/* Progress Ring */}
            <svg
              width={facePosition.width + 40}
              height={facePosition.height + 40}
              style={{ position: 'absolute', top: 0, left: 0 }}
            >
              {/* Background ring */}
              <circle
                cx={(facePosition.width + 40) / 2}
                cy={(facePosition.height + 40) / 2}
                r={(facePosition.width + 40) / 2 - 2}
                fill="none"
                stroke="rgba(255, 255, 255, 0.2)"
                strokeWidth="3"
              />
              {/* Progress ring */}
              <circle
                cx={(facePosition.width + 40) / 2}
                cy={(facePosition.height + 40) / 2}
                r={(facePosition.width + 40) / 2 - 2}
                fill="none"
                stroke="rgba(0, 220, 140, 0.8)"
                strokeWidth="3"
                strokeDasharray={`${2 * Math.PI * ((facePosition.width + 40) / 2 - 2)}`}
                strokeDashoffset={`${2 * Math.PI * ((facePosition.width + 40) / 2 - 2) * (1 - analysisProgress / 100)}`}
                strokeLinecap="round"
                transform={`rotate(-90 ${(facePosition.width + 40) / 2} ${(facePosition.height + 40) / 2})`}
                style={{ 
                  transition: 'stroke-dashoffset 0.3s cubic-bezier(0.4, 0.0, 0.2, 1)',
                  filter: 'drop-shadow(0 0 8px rgba(0, 220, 140, 0.4))'
                }}
              />
            </svg>
          </View>
        )}
        
        {/* Live Prediction Display */}
        {livePrediction !== null && !liveLoading && facePosition && videoRef.current && (
          <View style={{
            position: 'absolute',
            left: (videoRef.current.videoWidth || 640) - facePosition.x - 50, // Mirror the X position using video width
            top: facePosition.minY - 120, // Position above the face box (not on forehead)
            zIndex: 6,
            backgroundColor: 'rgba(0, 220, 140, 0.18)', // Same as Ready to capture badge
            borderRadius: 16,
            padding: 20,
            borderWidth: 1,
            borderColor: 'rgba(0, 220, 140, 0.45)', // Same as Ready to capture badge
            backdropFilter: 'blur(10px)', // Same as Ready to capture badge
            minWidth: 100,
            minHeight: 100,
            alignItems: 'center',
            justifyContent: 'center',
            filter: 'drop-shadow(0 0 10px rgba(0,220,140,0.35))' // Same glow as Ready to capture badge
          }}>
            <Text style={{
              color: '#E6EAF2',
              fontSize: 28,
              fontWeight: 'bold',
              textAlign: 'center',
              lineHeight: 32
            }}>
              {Math.round(livePrediction)}
            </Text>
            <Text style={{
              color: '#E6EAF2',
              fontSize: 12,
              textAlign: 'center',
              marginTop: 2,
              fontWeight: '600'
            }}>
              years old
            </Text>
          </View>
        )}
        
        {/* Results Button */}
        {showFullResultsButton && livePrediction !== null && (
          <View style={{
            position: 'absolute',
            bottom: 140,
            left: 0,
            right: 0,
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 6
          }}>
            {/* See Full Results Button */}
            <TouchableOpacity
              onPress={() => {
                // Capture current frame for the full results page
                const v = videoRef.current;
                if (v && canvasRef.current) {
                  const canvas = canvasRef.current;
                  canvas.width = v.videoWidth;
                  canvas.height = v.videoHeight;
                  const ctx = canvas.getContext('2d');
                  ctx.drawImage(v, 0, 0);
                  canvas.toBlob(async (blob) => {
                    if (blob) {
                      const imageUri = URL.createObjectURL(blob);
                      const compressedImage = await resizeImage(imageUri);
                      setSelectedImage(compressedImage);
                      setArCapturedImage(compressedImage);
                      closeWebCamera();
                      setSourceIsSelfie(true);
                      
                      // Use stored AR results instead of re-analyzing
                      if (arAnalysisResults) {
                        console.log('[LiveAR] Using stored AR results for full results page');
                        setResults(arAnalysisResults);
                        setCurrentStep(3); // Go directly to results
                      } else {
                        // Fallback to re-analysis if no stored results
                        console.log('[LiveAR] No stored results, falling back to re-analysis');
                        setCurrentStep(2);
                        analyzeFace(compressedImage);
                      }
                    }
                  }, 'image/jpeg', 0.8);
                }
              }}
              activeOpacity={0.85}
              style={{
                backgroundColor: 'rgba(0, 220, 140, 0.18)',
                borderColor: 'rgba(0, 220, 140, 0.45)',
                borderWidth: 1,
                paddingHorizontal: 20,
                paddingVertical: 12,
                borderRadius: 999,
                backdropFilter: 'blur(10px)',
                minWidth: 160,
                alignItems: 'center',
                boxShadow: '0 0 10px rgba(0,220,140,0.35)'
              }}
            >
              <Text style={{ color: '#E6EAF2', fontWeight: '700', fontSize: 16 }}>
                See Full Results
              </Text>
            </TouchableOpacity>
          </View>
        )}
        
        {/* Bottom overlay with buttons */}
        <View style={styles.cameraBottomOverlay}>
          <TouchableOpacity
            style={styles.cameraOverlayButton}
            onPress={closeWebCamera}
          >
            <Text style={styles.cameraButtonText}>Cancel</Text>
          </TouchableOpacity>
          
          {/* Only show camera button if no results are shown */}
          {!showFullResultsButton && (
            <TouchableOpacity
              style={[styles.cameraOverlayButton, styles.captureButton]}
              onPress={captureWebPhoto}
            >
              <View style={styles.captureButtonInner} />
            </TouchableOpacity>
          )}
          
          <View style={styles.cameraButtonSpacer} />
        </View>

        {/* Removed debug button */}
      </View>
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
          TrueAge uses deep learning models trained on tens of thousands of real people to estimate your biological age based on a picture of your face.
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

  const getModalContent = (type) => {
    switch(type) {
      case 'privacy':
        return (
          <>
            <Text category="h6" style={styles.modalTitle}>Privacy Policy</Text>
            <Text style={styles.modalText}>
              <Text style={styles.modalSectionTitle}>Data Collection{'\n'}</Text>
              TrueAge processes photos you upload solely for age estimation. We do not store your images permanently on our servers.{'\n\n'}
              
              <Text style={styles.modalSectionTitle}>How We Use Your Data{'\n'}</Text>
              • Photos are processed by AI models (Harvard FaceAge, DeepFace, OpenAI) to estimate age{'\n'}
              • Images are temporarily stored during processing and deleted afterward{'\n'}
              • No personal information is collected or stored{'\n\n'}
              
              <Text style={styles.modalSectionTitle}>Data Security{'\n'}</Text>
              All data transmission is encrypted. We use industry-standard security measures to protect your information during processing.{'\n\n'}
              
              <Text style={styles.modalSectionTitle}>Contact{'\n'}</Text>
              For privacy questions, contact: alexthorburnwinsor@gmail.com
            </Text>
          </>
        );
      case 'terms':
        return (
          <>
            <Text category="h6" style={styles.modalTitle}>Terms of Service</Text>
            <Text style={styles.modalText}>
              <Text style={styles.modalSectionTitle}>Acceptance of Terms{'\n'}</Text>
              By using TrueAge, you agree to these terms of service.{'\n\n'}
              
              <Text style={styles.modalSectionTitle}>Use of Service{'\n'}</Text>
              • TrueAge provides age estimation for entertainment and informational purposes{'\n'}
              • Results are estimates and not medical or legal advice{'\n'}
              • You must own or have permission to upload photos you submit{'\n\n'}
              
              <Text style={styles.modalSectionTitle}>Limitations{'\n'}</Text>
              Age estimates are provided "as is" without warranty. Results may vary and should not be used for official identification or medical purposes.{'\n\n'}
              
              <Text style={styles.modalSectionTitle}>Contact{'\n'}</Text>
              For questions about these terms, contact: alexthorburnwinsor@gmail.com
            </Text>
          </>
        );
      default:
        return null;
    }
  };

  const renderContentModal = () => {
    const content = (
      <Layout style={styles.infoModalCard}>
        <TouchableOpacity
          style={styles.closeIconContainer}
          onPress={() => setModalVisible(false)}
          activeOpacity={0.7}
        >
          <Icon name="close" fill="#888" style={styles.closeIcon} />
        </TouchableOpacity>
        {getModalContent(modalContent)}
      </Layout>
    );
    if (Platform.OS === 'web') {
      return modalVisible ? (
        <View style={styles.webModalOverlay}>
          {content}
        </View>
      ) : null;
    } else {
      return (
        <Modal
          visible={modalVisible}
          backdropStyle={{ backgroundColor: 'rgba(0,0,0,0.5)' }}
          onBackdropPress={() => setModalVisible(false)}
        >
          {content}
        </Modal>
      );
    }
  };

  // Step 1: Upload or Take Photo
  const renderStep1 = () => (
    <ScrollView contentContainerStyle={styles.mainContainer}>
      <Text category='h1' style={{ fontWeight: 'bold', textAlign: 'center', fontSize: 28, marginTop: 20, marginBottom: 10 }}>
        🧬 Find Out How Old You Look – Instantly
      </Text>
      <Text category='s1' style={styles.stepSubtitle}>
      Upload a photo to learn your biological and perceived age in seconds with advanced AI age recognition.
      </Text>
      {/* Upload/Take Photo UI below */}
      <View style={styles.demoImageContainer}>
        <Image 
          source={require('./assets/demoimage.webp')} 
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
          status='primary'
          appearance='filled'
          disabled={apiHealth?.models_loading && !apiHealth?.ready_for_analysis}
        >
          Take A Photo
        </Button>
        
        <Button
          style={styles.secondaryButton}
          onPress={pickImage}
          accessoryLeft={(props) => <Icon {...props} name='image-outline' fill='#E5E7EB'/>}
          size='large'
          status='basic'
          appearance='outline'
          disabled={apiHealth?.models_loading && !apiHealth?.ready_for_analysis}
        >
          <Text style={styles.secondaryButtonText}>Choose From Gallery</Text>
        </Button>
      </Layout>

      {/* FAQ Accordion */}
      <View style={{ width: '100%', maxWidth: MAIN_MAX_WIDTH, alignSelf: 'center', marginTop: 8, marginBottom: 8 }}>
        <FAQAccordion />
      </View>
      
      {/* Model status removed from UI - check console for status */}

      <AppFooter 
        onShowModal={(contentType) => { setModalContent(contentType); setModalVisible(true); }} 
        onShowInfo={() => setInfoVisible(true)}
      />
    </ScrollView>
  );

  // Step 2: Analyzing Photo
  const renderStep2 = () => (
    <ScrollView contentContainerStyle={styles.analyzingPageContent}>
      <Layout style={[styles.contentContainer, { minHeight: 0, maxHeight: undefined, justifyContent: 'center', alignItems: 'center' }]}> 
        {selectedImage && (
          <Layout style={[styles.analyzingImageContainer, { width: ANALYZE_IMAGE_SIZE, height: ANALYZE_IMAGE_SIZE }]}> 
            <Image 
              source={{ uri: selectedImage.uri }} 
              style={{
                width: ANALYZE_IMAGE_SIZE,
                height: ANALYZE_IMAGE_SIZE,
                borderRadius: 20,
                transform: [{ scaleX: sourceIsSelfie ? -1 : 1 }],
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
                  colors={['#4F8CFF', '#739CFF']}
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
      <AppFooter 
        onShowModal={(contentType) => { setModalContent(contentType); setModalVisible(true); }} 
        onShowInfo={() => setInfoVisible(true)}
      />
    </ScrollView>
  );

  // Step 3: Show Results
  const renderStep3 = () => (
    <ScrollView contentContainerStyle={styles.resultsScrollViewContent}>
      <Layout style={[styles.stepContainer, { maxWidth: MAIN_MAX_WIDTH, width: '100%', alignSelf: 'center', paddingLeft: 0, paddingRight: 0 }]}> 
        <Layout style={[styles.headerContainer, { paddingHorizontal: 0, paddingVertical: 0 }]}> 
          <Text category='h4' style={styles.stepTitle}>
            {(results && results.faces && ((results.override_used && results.faces.length > 0) || (results.faces.filter(face => face.confidence >= 0.9).length > 0)))
              ? '🎯 Analysis Results'
              : (results && results.message === 'No faces detected' ? 'No Face Detected' : 'No Clear Face Detected')}
          </Text>
          <Text category='s1' style={styles.stepSubtitle}>
            {(results && results.faces && ((results.override_used && results.faces.length > 0) || (results.faces.filter(face => face.confidence >= 0.9).length > 0)))
              ? 'Age estimation complete'
              : 'Try a front-facing photo in good lighting.'}
          </Text>
        </Layout>
        {results && results.faces && ((results.override_used && results.faces.length > 0) || results.faces.filter(face => face.confidence >= 0.9).length > 0) ? (
          (results.override_used ? results.faces : results.faces.filter(face => face.confidence >= 0.9)).map((face, index) => {
            const filteredFaces = results.override_used ? results.faces : results.faces.filter(face => face.confidence >= 0.9);
            const isSingleFace = filteredFaces.length === 1;
            
            // Compute consensus = average of all models that are shown
            const shownModelValues = [];
            if (face.age_harvard !== null && face.age_harvard !== undefined) shownModelValues.push(face.age_harvard);
            if (face.age_harvard_calibrated !== null && face.age_harvard_calibrated !== undefined) shownModelValues.push(face.age_harvard_calibrated);
            if (face.age_deepface !== null && face.age_deepface !== undefined) shownModelValues.push(face.age_deepface);
            if (face.age_chatgpt !== null && face.age_chatgpt !== undefined) shownModelValues.push(face.age_chatgpt);
            let consensus = null;
            if (shownModelValues.length > 0) {
              const sum = shownModelValues.reduce((acc, v) => acc + v, 0);
              consensus = sum / shownModelValues.length;
            }
            
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
                    <Layout style={styles.resultHeaderRight}>
                      {consensus !== null && (
                        <View style={styles.consensusPill}>
                          <Text style={styles.consensusPillText}>Consensus {Math.round(consensus)} yrs</Text>
                        </View>
                      )}
                      <Text category='c1' style={styles.confidenceText}>
                        {(face.confidence * 100).toFixed(1)}% confidence
                      </Text>
                    </Layout>
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
                          shadowColor: '#000',
                          shadowOffset: { width: 0, height: 4 },
                          shadowOpacity: 0.1,
                          shadowRadius: 8,
                          elevation: 5,
                          backgroundColor: '#151922',
                          justifyContent: 'center',
                          alignItems: 'center',
                          position: 'relative',
                          overflow: 'hidden',
                        }}>
                          <Image
                            source={{ uri: faceMeshOverlays[index] ? `data:image/jpeg;base64,${faceMeshOverlays[index]}` : `data:image/jpeg;base64,${face.face_crop_base64}` }}
                            style={{ width: '100%', height: '100%', borderRadius: 20, transform: [{ scaleX: sourceIsSelfie ? -1 : 1 }] }}
                            resizeMode="cover"
                          />
                          
                          {/* Age Estimate Badge Overlay */}
                          {consensus !== null && (
                            <View style={{
                              position: 'absolute',
                              top: 8,
                              right: 8,
                              backgroundColor: 'rgba(31, 60, 110, 0.83)', // Dark blue background
                              borderRadius: 12,
                              paddingHorizontal: 12,
                              paddingVertical: 8,
                              borderWidth: 2,
                              borderColor: '#4F8CFF', // Bright blue border
                              shadowColor: '#4F8CFF',
                              shadowOffset: { width: 0, height: 0 },
                              shadowOpacity: 0.6,
                              shadowRadius: 8,
                              elevation: 8,
                            }}>
                              <Text style={{
                                fontSize: 18,
                                fontWeight: 'bold',
                                color: '#FFFFFF', // White for age number
                                textAlign: 'center',
                              }}>
                                {Math.round(consensus)}
                              </Text>
                              <Text style={{
                                fontSize: 11,
                                color: '#FFFFFF', // White for "years"
                                textAlign: 'center',
                                marginTop: -2,
                                fontWeight: '500',
                              }}>
                                years
                              </Text>
                            </View>
                          )}
                        </View>
                      </Layout>
                    )}
                  </Layout>

                  {/* Age Estimation Results */}
                  <GlassPanel style={{
                    marginTop: 14,
                    marginBottom: 14,
                    borderRadius: 16,
                    padding: 14,
                    width: '100%',
                    alignSelf: 'center',
                    borderWidth: 1,
                    borderColor: 'rgba(255,255,255,0.06)'
                  }}>
                    <Layout style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'transparent', marginBottom: 8 }}>
                      <Text style={{ fontSize: 13, fontWeight: '700', color: '#E6EAF2', letterSpacing: 0.2 }}>AGE ESTIMATES</Text>
                      {isSingleFace && consensus !== null && (
                        <View style={styles.consensusPill}>
                          <Text style={styles.consensusPillText}>Average {Math.round(consensus)} yrs</Text>
                        </View>
                      )}
                    </Layout>
                    {/* Model rows */}
                    {(() => {
                      const modelRows = [];
                      if (face.age_harvard !== null && face.age_harvard !== undefined) modelRows.push({ key: 'harvard', value: face.age_harvard });
                      if (face.age_harvard_calibrated !== null && face.age_harvard_calibrated !== undefined) modelRows.push({ key: 'harvard_calibrated', value: face.age_harvard_calibrated });
                      if (face.age_deepface !== null && face.age_deepface !== undefined) modelRows.push({ key: 'deepface', value: face.age_deepface });
                      if (face.age_chatgpt !== null && face.age_chatgpt !== undefined) modelRows.push({ key: 'chatgpt', value: face.age_chatgpt });
                      return <>
                        {modelRows.map((row, i) => (
                          <Layout key={row.key} style={{ marginBottom: i === modelRows.length - 1 ? 0 : 12, backgroundColor: 'transparent' }}>
                            <Layout style={{ flexDirection: 'row', alignItems: 'center', backgroundColor: 'transparent', minHeight: 36 }}>
                              <Text style={{ fontSize: 18, marginRight: 8 }}>{MODEL_ICONS[row.key]}</Text>
                              <Layout style={{ flex: 1, backgroundColor: 'transparent' }}>
                                <Text style={{ fontWeight: '600', fontSize: 14, color: '#E6EAF2', marginBottom: 1 }}>{MODEL_LABELS[row.key]}</Text>
                                <Text style={{ fontSize: 11, color: '#9AA3AF', marginBottom: 2 }}>{MODEL_DESCRIPTIONS[row.key]}</Text>
                              </Layout>
                              <Text style={{ fontWeight: 'bold', fontSize: 15, color: '#E6EAF2' }}>{Math.round(row.value)}<Text style={{ fontSize: 12, color: '#9AA3AF' }}> yrs</Text></Text>
                            </Layout>
                            <View style={{ position: 'relative', height: 7, width: '100%', borderRadius: 4, backgroundColor: 'rgba(27,32,43,0.8)', overflow: 'hidden', marginTop: 2, marginBottom: 2 }}>
                              {(() => {
                                const min = 0, max = 100;
                                const percent = Math.max(0, Math.min(1, ((row.value - min) / (max - min))));
                                return percent > 0 ? (
                                  <LinearGradient
                                    colors={['#4F8CFF', '#739CFF']}
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
                              <View style={{ height: 1, backgroundColor: 'rgba(27,32,43,0.8)', marginTop: 12, marginBottom: 12, marginLeft: 0 }} />
                            )}
                          </Layout>
                        ))}
                      </>;
                    })()}
                  </GlassPanel>
                </Layout>

                {/* Age Factors */}
                {face.chatgpt_factors && (
                  <GlassPanel style={{
                    marginTop: 14,
                    marginBottom: 14,
                    borderRadius: 16,
                    padding: 14,
                    width: '100%',
                    alignSelf: 'center',
                    position: 'relative',
                    overflow: 'hidden',
                    borderWidth: 1,
                    borderColor: 'rgba(255,255,255,0.06)'
                  }}>
                    <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
                      <Text style={{ fontSize: 13, fontWeight: '700', color: '#E6EAF2', letterSpacing: 0.2, textAlign: 'left' }}>AGE FACTORS</Text>
                    </View>
                    
                    {/* Blurred content when not signed in AND auth is required */}
                    <View style={[
                      { 
                        filter: (user || !apiHealth?.require_auth) ? 'none' : 'blur(2px)',
                        opacity: (user || !apiHealth?.require_auth) ? 1 : 0.8,
                        transition: 'all 0.3s ease',
                      }
                    ]}>
                      {['skin_texture', 'skin_tone', 'hair', 'facial_volume'].map((factor, i, arr) => {
                        const f = face.chatgpt_factors[factor];
                        if (!f) return null;
                        const percent = Math.max(0, Math.min(1, ((f.age_rating - 0) / 100)));
                        return (
                          <Layout key={factor} style={{ marginBottom: i === arr.length - 1 ? 0 : 12, backgroundColor: 'transparent' }}>
                            <Layout style={{ flexDirection: 'row', alignItems: 'center', backgroundColor: 'transparent', minHeight: 36 }}>
                              <Text style={{ fontSize: 20, marginRight: 8 }}>{AGE_FACTOR_ICONS[factor]}</Text>
                              <Layout style={{ flex: 1, backgroundColor: 'transparent' }}>
                                <Text style={{ fontWeight: '600', fontSize: 14, color: '#E6EAF2', marginBottom: 1 }}>{AGE_FACTOR_LABELS[factor]}</Text>
                                <Text style={{ fontSize: 11, color: '#9AA3AF', marginBottom: 2 }}>{f.explanation}</Text>
                              </Layout>
                              <Text style={{ fontWeight: 'bold', fontSize: 15, color: '#E6EAF2', marginLeft: 10, minWidth: 32, textAlign: 'right' }}>{f.age_rating} <Text style={{ fontSize: 11, color: '#9AA3AF' }}>yrs</Text></Text>
                            </Layout>
                            <View style={{ position: 'relative', height: 7, width: '100%', borderRadius: 4, backgroundColor: 'rgba(27,32,43,0.8)', overflow: 'hidden', marginTop: 2, marginBottom: 2 }}>
                              <LinearGradient
                                colors={['#4F8CFF', '#739CFF']}
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
                              <View style={{ height: 1, backgroundColor: 'rgba(27,32,43,0.8)', marginTop: 12, marginBottom: 12, marginLeft: 0 }} />
                            )}
                          </Layout>
                        );
                      })}
                    </View>

                    {/* Sign Up Overlay when not signed in AND auth is required */}
                    {!user && apiHealth?.require_auth && (
                      <View style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(14, 17, 22, 0.6)',
                        backdropFilter: 'blur(4px)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderRadius: 12,
                      }}>
                        <View style={{
                          alignItems: 'center',
                          padding: 20,
                          maxWidth: 260,
                        }}>
                          <Text style={{
                            fontSize: 16,
                            fontWeight: '600',
                            color: '#E6EAF2',
                            textAlign: 'center',
                            marginBottom: 8,
                          }}>
                            🔒 Free account required
                          </Text>
                          <Text style={{
                            fontSize: 13,
                            color: '#9AA3AF',
                            textAlign: 'center',
                            marginBottom: 16,
                            lineHeight: 18,
                          }}>
                            Sign up free to see what factors make you look older and younger. It's completely free.
                          </Text>
                          {/* Email row */}
                          <View style={{ width: '100%', gap: 8, marginBottom: 12 }}>
                            <input
                              type="email"
                              placeholder="you@domain.com"
                              value={emailForLink}
                              onChange={(e) => setEmailForLink(e.target.value)}
                              ref={emailInputRef}
                              style={{
                                width: '100%',
                                padding: 12,
                                borderRadius: 12,
                                border: '1px solid rgba(255,255,255,0.08)',
                                background: 'rgba(21,25,34,0.8)',
                                color: '#E6EAF2',
                                outline: 'none',
                                boxSizing: 'border-box',
                              }}
                            />
                            <TouchableOpacity
                              disabled={emailSending}
                              style={{
                                width: '100%',
                                backgroundColor: '#4F8CFF',
                                opacity: emailSending ? 0.7 : 1,
                                paddingVertical: 12,
                                paddingHorizontal: 16,
                                borderRadius: 12,
                                alignItems: 'center',
                                justifyContent: 'center',
                                flexDirection: 'row',
                              }}
                              onPress={async () => {
                                setEmailSending(true);
                                const res = await sendMagicLink(emailForLink.trim());
                                setEmailSending(false);
                                setEmailSent(!!res.success);
                                if (!res.success) {
                                  Alert.alert('Error', res.error || 'Could not send link');
                                } else {
                                  Alert.alert('Check your email', 'We sent you a secure sign-in link.');
                                }
                              }}
                            >
                              {/* Email icon */}
                              {Platform.OS !== 'web' ? null : null}
                              <Ionicons name="mail-outline" size={16} color="#fff" style={{ marginRight: 8 }} />
                              <Text style={{ color: '#fff', fontWeight: '600', fontSize: 14, textAlign: 'center' }}>
                                {emailSent ? 'Link sent — check email' : (emailSending ? 'Sending…' : 'Continue with email')}
                              </Text>
                            </TouchableOpacity>
                          </View>
                          <TouchableOpacity
                            style={{
                              width: '100%',
                              backgroundColor: '#4F8CFF',
                              paddingVertical: 12,
                              paddingHorizontal: 16,
                              borderRadius: 12,
                              alignItems: 'center',
                              justifyContent: 'center',
                              flexDirection: 'row',
                              gap: 8,
                            }}
                            onPress={() => handleSignIn('google')}
                          >
                            <View style={{
                              width: 18,
                              height: 18,
                              borderRadius: 9,
                              backgroundColor: '#fff',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}>
                              <Text style={{ color: '#4285F4', fontWeight: '700', fontSize: 12 }}>G</Text>
                            </View>
                            <Text style={{
                              color: '#fff',
                              fontSize: 14,
                              fontWeight: '600',
                              textAlign: 'center',
                            }}>
                              Continue with Google
                            </Text>
                          </TouchableOpacity>
                          {/* Login link */}
                          <View style={{ marginTop: 10, alignItems: 'center', width: '100%' }}>
                            <Text style={{ color: '#9AA3AF', fontSize: 12 }}>
                              Already have an account?{' '}
                              <Text
                                onPress={async () => {
                                  const email = (emailForLink || '').trim();
                                  if (Platform.OS === 'web') {
                                    if (email.length > 0) {
                                      try {
                                        setEmailSending(true);
                                        const res = await sendMagicLink(email);
                                        setEmailSending(false);
                                        setEmailSent(!!res.success);
                                        if (!res.success) {
                                          Alert.alert('Error', res.error || 'Could not send link');
                                        } else {
                                          Alert.alert('Check your email', 'We sent you a secure sign-in link.');
                                        }
                                      } catch (e) {
                                        setEmailSending(false);
                                        Alert.alert('Error', 'Could not send link');
                                      }
                                    } else {
                                      // Focus the email field for convenience
                                      try { emailInputRef?.current?.focus && emailInputRef.current.focus(); } catch {}
                                      try { showToast && showToast('Enter your email to get a login link.'); } catch {}
                                    }
                                  } else {
                                    // On native, fall back to Google sign-in
                                    handleSignIn('google');
                                  }
                                }}
                                style={{ color: '#4F8CFF', fontSize: 12, fontWeight: '600', textDecorationLine: 'underline' }}
                              >
                                Log in
                              </Text>
                            </Text>
                          </View>
                        </View>
                      </View>
                    )}
                  </GlassPanel>
                )}
              </Layout>
            </Card>
          );
          })
        ) : (null)}
        <Layout style={[styles.resultsActions, { maxWidth: MAIN_MAX_WIDTH, width: '100%', alignSelf: 'center' }]}> 
          {results && ((results.override_used && results.faces && results.faces.length > 0) || (results.faces && results.faces.filter(face => face.confidence >= 0.9).length > 0)) ? (
            <Button
              style={styles.shareButton}
              onPress={shareResults}
              accessoryLeft={ShareIcon}
              status='primary'
              appearance='filled'
            >
              Share Results
            </Button>
          ) : (results && results.message !== 'No faces detected') ? (
            <Button
              style={styles.shareButton}
              onPress={() => { setCurrentStep(2); analyzeFace(selectedImage, { overrideLowConfidence: true }); }}
              accessoryLeft={ShieldAlertIcon}
              status='primary'
              appearance='filled'
            >
              Analyze anyway
            </Button>
          ) : null}
          <Button
            style={styles.secondaryButton}
            onPress={() => setCurrentStep(1)}
            accessoryLeft={ArrowBackIcon}
            status='primary'
            appearance='outline'
          >
            Try Another Photo
          </Button>
        </Layout>
      </Layout>
      <AppFooter 
        onShowModal={(contentType) => { setModalContent(contentType); setModalVisible(true); }} 
        onShowInfo={() => setInfoVisible(true)}
      />
    </ScrollView>
  );

  // renderApiStatus function removed - model status now logged to console

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" backgroundColor="#0E1116" />
      <AppHeader 
        onShowInfo={() => setInfoVisible(true)} 
        user={user}
        onSignIn={handleSignIn}
        onSignOut={handleSignOut}
      />
      <Layout style={styles.fullScreen}>
        <Suspense fallback={<LoadingSpinner message="Loading TrueAge..." />}>
        {currentStep === 1 && renderStep1()}
        {currentStep === 2 && renderStep2()}
        {currentStep === 3 && renderStep3()}
        </Suspense>
      </Layout>
      {renderInfoModal()}
      {renderContentModal()}
      {Platform.OS === 'web' && toast.visible && (
        <Animated.View style={[styles.toastOverlay, { opacity: toastOpacity }] }>
          <GlassPanel style={styles.toastCard}>
            <Text style={styles.toastText}>{toast.text}</Text>
          </GlassPanel>
        </Animated.View>
      )}
    </SafeAreaView>
  );
}

export default function App() {
  useEffect(() => {
    Asset.loadAsync(require('./assets/logo.png'));
  }, []);
  return (
    <Fragment>
      <IconRegistry icons={EvaIconsPack} />
      <ApplicationProvider {...eva} theme={{ ...eva.dark, ...customDark }}>
        <AppContent />
      </ApplicationProvider>
    </Fragment>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0E1116',
  },
  fullScreen: {
    flex: 1,
    backgroundColor: '#0E1116',
  },
  stepContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: '#0E1116',
  },
  headerContainer: {
    alignItems: 'center',
    marginBottom: 12,
    paddingTop: 6,
  },
  stepTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#E6EAF2',
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 4,
  },
  stepSubtitle: {
    fontSize: 16,
    color: '#9AA3AF',
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
    marginTop: 40,
    marginBottom: 52,
    marginLeft: 40,
    marginRight: 40,
  },
  demoImage: {
    width: Math.min(width * 0.65, 300),
    height: Math.min(width * 0.4, 300),
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
    backgroundColor: '#151922',
    borderRadius: 24,
    borderWidth: 1,
    borderColor: '#2A3650',
    marginBottom: 10,
    paddingVertical: 14,
  },
  secondaryButtonText: {
    color: '#E5E7EB',
    fontSize: 16,
    fontWeight: '600',
  },
  facialRegionsToggle: {
    alignItems: 'center',
    marginBottom: 12,
  },
  toggleButton: {
    backgroundColor: '#151922',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: '#2a2f3a',
  },
  toggleButtonActive: {
    backgroundColor: '#4f8cff',
    borderColor: '#4f8cff',
  },
  toggleButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#9AA3AF',
  },
  toggleButtonTextActive: {
    color: '#ffffff',
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
    // justifyContent: 'center', // Removed to allow scan line to start at the top
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
    borderColor: '#4f8cff',
    opacity: 0.8,
    borderTopLeftRadius: 8,
},
  scanCornerTopRight: {
    position: 'absolute',
    width: 18,
    height: 18,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderColor: '#4f8cff',
    opacity: 0.8,
    borderTopRightRadius: 8,
},
  scanCornerBottomLeft: {
    position: 'absolute',
    width: 18,
    height: 18,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#4f8cff',
    opacity: 0.8,
    borderBottomLeftRadius: 8,
},
  scanCornerBottomRight: {
    position: 'absolute',
    width: 18,
    height: 18,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderColor: '#4f8cff',
    opacity: 0.8,
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
    marginBottom: 12,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
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
    marginBottom: 10,
  },
  resultHeaderRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  resultTitle: {
    fontWeight: 'bold',
  },
  confidenceText: {
    opacity: 0.7,
  },
  consensusPill: {
    borderRadius: 999,
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: 'rgba(79,140,255,0.28)',
    backgroundColor: 'rgba(46,90,199,0.18)',
  },
  consensusPillText: {
    color: '#E6EAF2',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
  resultContent: {
    gap: 14,
  },
  analysisHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: '#151922',
    borderRadius: 8,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#2a2f3a',
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
    gap: 14,
  },
  faceDetectionArea: {
    gap: 10,
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
    marginVertical: 10,
    backgroundColor: 'transparent',
    position: 'relative',
  },
  faceCropImage: {
    width: 200,
    height: 200,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#1B202B',
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
    gap: 10,
  },
  ageResultCard: {
    backgroundColor: '#151922',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2a2f3a',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
    marginBottom: 10,
    width: '100%',
    alignSelf: 'center',
    paddingVertical: 12,
    paddingHorizontal: 14,
    minHeight: 56,
  },
  ageResultRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    paddingHorizontal: 2,
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
    color: '#E6EAF2',
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
    color: '#9AA3AF',
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
    color: '#9AA3AF',
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
    color: '#9AA3AF',
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
    backgroundColor: '#151922',
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2f3a',
  },
  summaryAge: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#E6EAF2',
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
    backgroundColor: '#2a2f3a',
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
    borderRadius: 24,
    marginTop: 10,
    marginBottom: 8,
    paddingVertical: 14,
    borderWidth: 0,
    borderColor: 'transparent',
    // Remove focus/outline ring on web
    outlineStyle: 'none',
    outlineWidth: 0,
    outlineColor: 'transparent',
  },
  webCameraCard: {
    flex: 1,
    margin: 20,
    borderWidth: 0,
    borderColor: 'transparent',
    borderStyle: 'none',
    backgroundColor: '#151922',
    maxWidth: 500,
    width: '100%',
    alignSelf: 'center',
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
    transform: 'scaleX(-1)', // Mirror the video horizontally
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
    backgroundColor: '#151922',
    borderWidth: 1,
    borderColor: '#2a2f3a',
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
    color: '#E6EAF2',
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 15,
  },
  confidenceValue: {
    fontSize: 12,
    color: '#3366FF',
    fontWeight: '600',
  },
  headerLogo: {
    width: 32,
    height: 32,
    marginRight: 10,
    borderRadius: 8,
    resizeMode: 'contain',
  },
  headerTitle: {
    fontWeight: 'bold',
    color: '#E6EAF2',
    fontSize: 20,
    letterSpacing: 0.5,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
  },
  headerNav: {
    backgroundColor: '#0E1116',
    borderBottomWidth: 1,
    borderBottomColor: '#1B202B',
    minHeight: 60,
    paddingHorizontal: 8,
  },
  mainContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-start',
    padding: 24,
    maxWidth: MAIN_MAX_WIDTH,
    width: '100%',
    alignSelf: 'center',
  },
  resultsScrollViewContent: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'flex-start',
    minHeight: height,
    backgroundColor: '#0E1116',
    width: '100%',
  },
  analyzingPageContent: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: height - 160, // Account for header height, safe areas, and iOS-specific elements
    backgroundColor: '#0E1116',
    width: '100%',
  },
  footer: {
    paddingVertical: 24,
    paddingHorizontal: 20,
  },
  footerContent: {
    maxWidth: MAIN_MAX_WIDTH,
    width: '100%',
    alignSelf: 'center',
  },
  footerLinks: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginBottom: 12,
  },
  footerLink: {
    color: '#4F8CFF',
    fontSize: 14,
    fontWeight: '500',
  },
  footerSeparator: {
    color: '#9AA3AF',
    marginHorizontal: 8,
  },
  footerCopyright: {
    color: '#9AA3AF',
    fontSize: 12,
    textAlign: 'center',
  },
  modalTitle: {
    fontWeight: 'bold',
    marginBottom: 16,
    fontSize: 18,
  },
  modalText: {
    fontSize: 14,
    lineHeight: 20,
    color: '#E6EAF2',
  },
  modalSectionTitle: {
    fontWeight: 'bold',
    fontSize: 15,
  },
  fullScreenCamera: {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100dvh', // Dynamic viewport height for mobile browsers
    backgroundColor: '#000',
    zIndex: 9999,
  },
  fullScreenVideo: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    transform: 'scaleX(-1)', // Mirror the video horizontally
  },
  cameraTopOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 20,
    background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)',
    alignItems: 'center',
  },
  cameraTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  cameraBottomOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingTop: 40,
    paddingBottom: Math.max(50, typeof window !== 'undefined' && window.screen ? window.screen.height * 0.08 : 50),
    paddingHorizontal: 40,
    background: 'linear-gradient(to top, rgba(0,0,0,0.7), transparent)',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  cameraOverlayButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 25,
    backgroundColor: 'rgba(255,255,255,0.2)',
    backdropFilter: 'blur(10px)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  cameraButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 0,
    paddingHorizontal: 0,
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
  cameraButtonSpacer: {
    width: 80, // Same width as capture button for centering
  },
  faceOutlineContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  faceOutlineOval: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: FACE_WIDTH,
    height: FACE_HEIGHT,
    borderRadius: FACE_WIDTH * 0.5,
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: 'rgba(0, 255, 255, 0.55)',
    shadowColor: 'rgba(255, 255, 255, 0.5)',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 15,
  },
  instructionContainer: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 20,
    marginTop: 10,
  },
  instructionText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
    textShadowColor: 'rgba(0, 0, 0, 0.7)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  toastOverlay: {
    position: 'fixed',
    left: 0,
    right: 0,
    bottom: 24,
    display: 'flex',
    alignItems: 'center',
    zIndex: 9999,
    pointerEvents: 'none',
  },
  toastCard: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    backgroundColor: 'rgba(28,34,44,0.55)',
    boxShadow: '0 6px 24px rgba(0,0,0,0.25)',
  },
  toastText: {
    color: '#E6EAF2',
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
});

