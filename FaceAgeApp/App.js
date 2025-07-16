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
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Ionicons } from '@expo/vector-icons';

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
} from '@ui-kitten/components';

const { width, height } = Dimensions.get('window');

// API Configuration
const getApiBaseUrl = () => {
  // Check if we're in development or production
  if (typeof window !== 'undefined' && window.location) {
    // Web environment
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      // Local development - use /api prefix for unified deployment
      return 'http://localhost:8000/api';
    } else {
      // Production - use relative URLs since API is served from same domain
      return '/api';
    }
  } else {
    // React Native mobile environment
    return 'http://192.168.96.130:8000/api';
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

      const response = await fetch(`${API_BASE_URL}/analyze-face`, {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);

      if (response.ok) {
        const data = await response.json();
        console.log('Analysis results:', data);
        setResults(data);
        
        // Add minimum delay to show scanning animation
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
      Alert.alert('Error', `Network error: ${error.message}`);
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

  // Step 1: Upload or Take Photo
  const renderStep1 = () => (
    <Layout style={styles.stepContainer}>
      <Layout style={styles.headerContainer}>
        <Text category='h4' style={styles.stepTitle}>📸 Upload or Take Photo</Text>
        <Text category='s1' style={styles.stepSubtitle}>
          Choose a clear photo with visible face(s)
        </Text>
      </Layout>

      <Layout style={styles.contentContainer}>
        <Layout style={styles.apiStatusContainer}>
          {apiHealth ? (
            <Text category='c1' style={[
              styles.apiStatus,
              apiHealth.status === 'healthy' ? styles.apiConnected : styles.apiDisconnected
            ]}>
              {apiHealth.status === 'healthy' ? '✅ API Connected' : '❌ API Disconnected'}
            </Text>
          ) : (
            <Text category='c1' style={styles.apiStatus}>🔄 Checking API...</Text>
          )}
        </Layout>

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
      </Layout>
    </Layout>
  );

  // Step 2: Analyzing Photo
  const renderStep2 = () => (
    <Layout style={styles.stepContainer}>
      <Layout style={styles.headerContainer}>
        <Text category='h4' style={styles.stepTitle}>🔍 Analyzing Photo</Text>
        <Text category='s1' style={styles.stepSubtitle}>
          Please wait while we analyze your photo...
        </Text>
      </Layout>

      <Layout style={styles.contentContainer}>
        {selectedImage && (
          <Layout style={styles.analyzingImageContainer}>
            <Image 
              source={{ uri: selectedImage.uri }} 
              style={styles.analyzingImage}
              resizeMode="contain"
            />
            <View style={styles.scanOverlay}>
              <Animated.View style={[
                styles.scanLine,
                {
                  opacity: scanLineOpacity,
                  transform: [{
                    translateY: scanLinePosition.interpolate({
                      inputRange: [0, 1],
                      outputRange: [-(width * 0.6) / 2, (width * 0.6) / 2],
                    })
                  }]
                }
              ]} />
              
              {/* Corner brackets for scanning effect */}
              <View style={styles.scanCornerTopLeft} />
              <View style={styles.scanCornerTopRight} />
              <View style={styles.scanCornerBottomLeft} />
              <View style={styles.scanCornerBottomRight} />
            </View>
          </Layout>
        )}

        <Layout style={styles.loadingContainer}>
          <Spinner size='large' />
          <Text category='h6' style={styles.loadingText}>
            Detecting faces and analyzing age...
          </Text>
          <Text category='c1' style={styles.loadingSubtext}>
            Using Harvard FaceAge + DeepFace models
          </Text>
        </Layout>
      </Layout>
    </Layout>
  );

  // Step 3: Show Results
  const renderStep3 = () => (
    <Layout style={styles.stepContainer}>
      <Layout style={styles.headerContainer}>
        <Text category='h4' style={styles.stepTitle}>🎯 Analysis Results</Text>
        <Text category='s1' style={styles.stepSubtitle}>
          Age estimation complete
        </Text>
      </Layout>

      <ScrollView style={styles.resultsScrollView}>
        {results && results.faces && results.faces.filter(face => face.confidence >= 0.9).length > 0 ? (
          results.faces.filter(face => face.confidence >= 0.9).map((face, index) => (
            <Card key={index} style={styles.resultCard}>
              <Layout style={styles.resultHeader}>
                <Text category='h6' style={styles.resultTitle}>
                  Face {index + 1}
                </Text>
                <Text category='c1' style={styles.confidenceText}>
                  {(face.confidence * 100).toFixed(1)}% confidence
                </Text>
              </Layout>
              
              <Layout style={styles.resultContent}>
                {/* Face crop image */}
                {face.face_crop_base64 && (
                  <Layout style={styles.faceCropContainer}>
                    <Image
                      source={{ uri: `data:image/jpeg;base64,${face.face_crop_base64}` }}
                      style={styles.faceCropImage}
                      resizeMode="cover"
                    />
                  </Layout>
                )}
                
                <Layout style={styles.ageResultsContainer}>
                  {face.harvard_age && (
                    <Layout style={styles.ageResult}>
                      <Text category='label' style={styles.ageLabel}>🎯 Harvard Age</Text>
                      <Text category='h5' style={styles.ageValue}>
                        {face.harvard_age.toFixed(1)} years
                      </Text>
                    </Layout>
                  )}
                  
                  {face.deepface_age && (
                    <Layout style={styles.ageResult}>
                      <Text category='label' style={styles.ageLabel}>🤖 DeepFace Age</Text>
                      <Text category='h5' style={styles.ageValue}>
                        {face.deepface_age.toFixed(1)} years
                      </Text>
                    </Layout>
                  )}
                </Layout>
                
                {/* Age Category */}
                <Layout style={styles.categoryContainer}>
                  {(() => {
                    const primaryAge = face.harvard_age || face.deepface_age;
                    if (primaryAge < 30) {
                      return (
                        <Text category='h6' style={[styles.categoryText, styles.youngText]}>
                          😊 Young
                        </Text>
                      );
                    } else if (primaryAge < 50) {
                      return (
                        <Text category='h6' style={[styles.categoryText, styles.adultText]}>
                          😐 Adult
                        </Text>
                      );
                    } else {
                      return (
                        <Text category='h6' style={[styles.categoryText, styles.seniorText]}>
                          👴 Senior
                        </Text>
                      );
                    }
                  })()}
                </Layout>
              </Layout>
            </Card>
          ))
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

      <Layout style={styles.resultsActions}>
        <Button
          style={styles.secondaryButton}
          onPress={() => setCurrentStep(1)}
          accessoryLeft={ArrowBackIcon}
          status='basic'
        >
          Try Another Photo
        </Button>
        
        <Button
          style={styles.primaryButton}
          onPress={resetApp}
          accessoryLeft={RefreshIcon}
        >
          Start Over
        </Button>
      </Layout>
    </Layout>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="auto" />
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
    backgroundColor: '#f8f9fa',
  },
  fullScreen: {
    flex: 1,
  },
  stepContainer: {
    flex: 1,
    padding: 20,
  },
  headerContainer: {
    alignItems: 'center',
    marginBottom: 30,
    paddingTop: 20,
  },
  stepTitle: {
    textAlign: 'center',
    marginBottom: 10,
    fontWeight: 'bold',
  },
  stepSubtitle: {
    textAlign: 'center',
    opacity: 0.7,
  },
  contentContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  apiStatusContainer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  apiStatus: {
    fontSize: 14,
    fontWeight: '600',
  },
  apiConnected: {
    color: '#4CAF50',
  },
  apiDisconnected: {
    color: '#f44336',
  },
  buttonContainer: {
    gap: 20,
  },
  primaryButton: {
    borderRadius: 25,
    paddingVertical: 15,
  },
  secondaryButton: {
    borderRadius: 25,
    paddingVertical: 15,
  },
  analyzingImageContainer: {
    alignItems: 'center',
    marginBottom: 40,
    position: 'relative',
  },
  analyzingImage: {
    width: width * 0.8,
    height: width * 0.6,
    borderRadius: 15,
  },
  scanOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  scanLine: {
    width: '85%',
    height: 4,
    backgroundColor: '#00E676',
    borderRadius: 2,
    shadowColor: '#00E676',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 8,
    elevation: 5,
  },
  scanCornerTopLeft: {
    position: 'absolute',
    top: 10,
    left: 10,
    width: 20,
    height: 20,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#00E676',
  },
  scanCornerTopRight: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 20,
    height: 20,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderColor: '#00E676',
  },
  scanCornerBottomLeft: {
    position: 'absolute',
    bottom: 10,
    left: 10,
    width: 20,
    height: 20,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#00E676',
  },
  scanCornerBottomRight: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    width: 20,
    height: 20,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderColor: '#00E676',
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
    gap: 15,
  },
  faceCropContainer: {
    alignItems: 'center',
    marginBottom: 15,
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 15,
    borderWidth: 1,
    borderColor: '#e9ecef',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  faceCropImage: {
    width: 100,
    height: 120,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: '#2196F3',
    backgroundColor: '#fff',
  },
  ageResultsContainer: {
    gap: 15,
  },
  ageResult: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 15,
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
  },
  ageLabel: {
    fontSize: 16,
    fontWeight: '600',
  },
  ageValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  categoryContainer: {
    alignItems: 'center',
    paddingVertical: 15,
  },
  categoryText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  youngText: {
    color: '#4CAF50',
  },
  adultText: {
    color: '#FF9800',
  },
  seniorText: {
    color: '#9C27B0',
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
    flexDirection: 'row',
    gap: 15,
    marginTop: 20,
  },
  webCameraCard: {
    flex: 1,
    margin: 20,
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
});
