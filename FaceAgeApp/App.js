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

const { width } = Dimensions.get('window');

// API Configuration
const getApiBaseUrl = () => {
  // Check if we're in development or production
  if (typeof window !== 'undefined' && window.location) {
    // Web environment
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      // Local development
      return 'http://192.168.96.130:8000';
    } else {
      // Production - use relative URLs since API is served from same domain
      return '/api';
    }
  } else {
    // React Native mobile environment
    return 'http://192.168.96.130:8000';
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

function AppContent() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [showWebCamera, setShowWebCamera] = useState(false);
  const [webCameraStream, setWebCameraStream] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Camera permissions are handled by ImagePicker internally

  // Check API health
  useEffect(() => {
    checkApiHealth();
  }, []);

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
      setApiHealth({ status: 'error', message: 'Connection failed' });
    }
  };

  const resizeImage = async (uri) => {
    try {
      const resized = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 800 } }],
        {
          compress: 0.8,
          format: ImageManipulator.SaveFormat.JPEG,
        }
      );
      return resized;
    } catch (error) {
      console.error('Image resize error:', error);
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
        setResults(null);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image');
    }
  };

  const takePhoto = async () => {
    try {
      if (Platform.OS === 'web') {
        // Use web camera for browsers
        setShowWebCamera(true);
      } else {
        // Use ImagePicker for mobile
        const result = await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: true,
          aspect: [4, 3],
          quality: 1,
        });

        if (!result.canceled) {
          const compressedImage = await resizeImage(result.assets[0].uri);
          setSelectedImage(compressedImage);
          setResults(null);
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
        
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw video frame to canvas
        context.drawImage(video, 0, 0);
        
        // Convert to blob
        canvas.toBlob(async (blob) => {
          const imageUri = URL.createObjectURL(blob);
          const compressedImage = await resizeImage(imageUri);
          setSelectedImage(compressedImage);
          setResults(null);
          closeWebCamera();
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

  const analyzeFace = async () => {
    if (!selectedImage) return;

    setLoading(true);
    try {
      const formData = new FormData();
      
      // Check if we're in web environment (React Native Web)
      if (typeof selectedImage.uri === 'string' && 
          (selectedImage.uri.startsWith('data:') || selectedImage.uri.startsWith('blob:'))) {
        // Handle web environment (data URLs or blob URLs)
        const response = await fetch(selectedImage.uri);
        const blob = await response.blob();
        formData.append('file', blob, 'image.jpg');
      } else {
        // For React Native (mobile) - need to create proper file object
        const fileInfo = {
          uri: selectedImage.uri,
          type: 'image/jpeg',
          name: 'image.jpg',
        };
        
        // Use the file object directly for React Native
        formData.append('file', fileInfo);
      }

      console.log('Sending request to:', `${API_BASE_URL}/analyze-face`);
      console.log('FormData created with file:', selectedImage.uri);

      const response = await fetch(`${API_BASE_URL}/analyze-face`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - let the browser set it with boundary
      });

      console.log('Response status:', response.status);

      if (response.ok) {
        const data = await response.json();
        console.log('Analysis results:', data);
        setResults(data);
      } else {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        Alert.alert('Error', `Analysis failed: ${response.status}`);
      }
    } catch (error) {
      console.error('Network error:', error);
      Alert.alert('Error', `Network error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedImage(null);
    setResults(null);
    closeWebCamera();
  };

  // Permission checks removed - ImagePicker handles permissions internally

  // Web camera view for browsers
  if (showWebCamera && Platform.OS === 'web') {
    return (
      <SafeAreaView style={styles.safeArea}>
        <Layout style={styles.container}>
          <Layout style={styles.webCameraContainer}>
          <video
            ref={videoRef}
            style={styles.webCameraVideo}
            autoPlay
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            style={{ display: 'none' }}
          />
          <Layout style={styles.webCameraControls}>
            <Button
              style={styles.button}
              onPress={closeWebCamera}
              appearance='ghost'
            >
              Cancel
            </Button>
            <Button
              style={styles.captureButton}
              onPress={captureWebPhoto}
              accessoryLeft={CameraIcon}
            >
              Capture
            </Button>
          </Layout>
        </Layout>
      </Layout>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <Layout style={styles.container}>
        <StatusBar style="dark" backgroundColor="white" translucent={false} />
        
        <ScrollView style={styles.scrollContainer}>
        {/* Header */}
        <Card style={styles.headerCard}>
          <Layout style={styles.header}>
            <Text category='h4' style={styles.title}>📱 Facial Age Estimator</Text>
            <Text category='s1' style={styles.subtitle}>
              Estimate biological age from facial features
            </Text>
            <Divider style={styles.divider} />
            <Layout style={styles.healthContainer}>
              <Avatar 
                size='tiny' 
                style={[
                  styles.healthIndicator,
                  { backgroundColor: apiHealth?.status === 'healthy' ? '#4CAF50' : '#F44336' }
                ]}
              />
              <Text category='c1' style={styles.healthText}>
                API: {apiHealth?.status === 'healthy' ? 'Connected' : 'Disconnected'}
              </Text>
            </Layout>
          </Layout>
        </Card>

        {/* Action Buttons */}
        <Card style={styles.card}>
          <Text category='h6' style={styles.sectionTitle}>📸 Capture Photo</Text>
          <Layout style={styles.buttonContainer}>
            <Button
              style={styles.button}
              onPress={takePhoto}
              accessoryLeft={CameraIcon}
            >
              Take Photo
            </Button>
            <Button
              style={styles.button}
              onPress={pickImage}
              accessoryLeft={ImageIcon}
            >
              Choose Photo
            </Button>
          </Layout>
        </Card>

        {/* Selected Image */}
        {selectedImage && (
          <Card style={styles.card}>
            <Text category='h6' style={styles.sectionTitle}>📷 Selected Image</Text>
            <Image source={{ uri: selectedImage.uri }} style={styles.image} />
            
            <Layout style={styles.buttonContainer}>
              <Button
                style={[styles.button, styles.analyzeButton]}
                onPress={analyzeFace}
                disabled={loading}
                accessoryLeft={loading ? undefined : AnalyticsIcon}
              >
                {loading ? <Spinner size='small' /> : 'Analyze Face'}
              </Button>
              
              <Button
                style={[styles.button, styles.resetButton]}
                onPress={resetApp}
                accessoryLeft={RefreshIcon}
              >
                Reset
              </Button>
            </Layout>
          </Card>
        )}

        {/* Results */}
        {results && (
          <Card style={styles.card}>
            <Text category='h6' style={styles.sectionTitle}>🎯 Analysis Results</Text>
            
            {/* Face Detection Results */}
            {results.faces && results.faces.length > 0 ? (
              results.faces.map((face, index) => (
                <Card key={index} style={styles.resultCard}>
                  <Layout style={styles.resultContainer}>
                    <Text category='h6' style={styles.resultTitle}>
                      Face {index + 1}
                    </Text>
                    
                    {face.harvard_age && (
                      <Layout style={styles.metricContainer}>
                        <Text category='label' style={styles.metricLabel}>🎯 Harvard Age</Text>
                        <Text category='h6' style={styles.metricValue}>
                          {face.harvard_age.toFixed(1)} years
                        </Text>
                      </Layout>
                    )}
                    
                    {face.deepface_age && (
                      <Layout style={styles.metricContainer}>
                        <Text category='label' style={styles.metricLabel}>🤖 DeepFace Age</Text>
                        <Text category='h6' style={styles.metricValue}>
                          {face.deepface_age.toFixed(1)} years
                        </Text>
                      </Layout>
                    )}
                    
                    <Layout style={styles.metricContainer}>
                      <Text category='label' style={styles.metricLabel}>Face Detection</Text>
                      <Text category='h6' style={styles.metricValue}>
                        {(face.confidence * 100).toFixed(1)}%
                      </Text>
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
              <Text category='s1' style={styles.noResultsText}>
                No faces detected in the image
              </Text>
            )}
          </Card>
        )}
      </ScrollView>
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
  safeArea: {
    flex: 1,
    backgroundColor: 'white',
  },
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContainer: {
    padding: 15,
  },
  headerCard: {
    marginBottom: 15,
  },
  header: {
    alignItems: 'center',
  },
  title: {
    textAlign: 'center',
    marginBottom: 5,
  },
  subtitle: {
    textAlign: 'center',
    marginBottom: 15,
  },
  divider: {
    marginVertical: 15,
  },
  healthContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  healthIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  healthText: {
    fontSize: 14,
  },
  card: {
    marginBottom: 15,
  },
  sectionTitle: {
    textAlign: 'center',
    marginBottom: 15,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 15,
  },
  button: {
    minWidth: 120,
    marginHorizontal: 5,
  },
  analyzeButton: {
    minWidth: 120,
    marginHorizontal: 5,
  },
  resetButton: {
    minWidth: 120,
    marginHorizontal: 5,
  },
  image: {
    width: '100%',
    height: (width - 100) * 0.75,
    borderRadius: 8,
    alignSelf: 'center',
    marginBottom: 15,
  },
  // Web camera styles
  webCameraContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  webCameraVideo: {
    width: '100%',
    maxWidth: 640,
    height: 'auto',
    borderRadius: 10,
  },
  webCameraControls: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
  resultContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  resultText: {
    fontSize: 16,
    color: '#666',
    marginTop: 10,
  },
  faceResult: {
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
  },
  faceTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    textAlign: 'center',
  },
  faceImage: {
    width: 150,
    height: 150,
    borderRadius: 8,
    alignSelf: 'center',
    marginBottom: 15,
  },
  ageContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 15,
  },
  ageResult: {
    alignItems: 'center',
  },
  ageLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  ageValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  categoryBadge: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'center',
  },
  categoryText: {
    color: 'white',
    fontWeight: 'bold',
  },
  modelInfo: {
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
  },
  modelTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  modelDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  warningContainer: {
    backgroundColor: '#FFF3CD',
    borderRadius: 8,
    padding: 15,
    borderLeftWidth: 4,
    borderLeftColor: '#FF9800',
  },
  warningText: {
    fontSize: 14,
    color: '#856404',
    lineHeight: 20,
  },
  // New styles for UI Kitten
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
  },
  errorText: {
    textAlign: 'center',
  },
  errorSubtext: {
    textAlign: 'center',
    marginTop: 5,
  },
  cameraHeader: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 1,
  },
  resultCard: {
    marginBottom: 15,
  },
  metricContainer: {
    alignItems: 'center',
    marginBottom: 10,
  },
  metricLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  categoryContainer: {
    alignSelf: 'center',
    marginTop: 15,
  },
  youngText: {
    backgroundColor: '#4CAF50',
  },
  adultText: {
    backgroundColor: '#2196F3',
  },
  seniorText: {
    backgroundColor: '#FF9800',
  },
  noResultsText: {
    textAlign: 'center',
    color: '#666',
    marginTop: 20,
  },
});
