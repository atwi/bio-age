import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Alert,
  ScrollView,
  SafeAreaView,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Camera } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

// API Configuration
const API_BASE_URL = 'http://localhost:8000'; // Change this to your deployed API URL

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraRef, setCameraRef] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  // Check camera permissions
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  // Check API health
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        timeout: 5000,
      });
      
      if (response.ok) {
        const data = await response.json();
        setApiHealth(data);
      } else {
        setApiHealth({ status: 'error', message: 'API not responding' });
      }
    } catch (error) {
      setApiHealth({ status: 'error', message: error.message });
    }
  };

  const takePicture = async () => {
    if (cameraRef) {
      try {
        const photo = await cameraRef.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        
        // Resize image for better performance
        const resizedPhoto = await ImageManipulator.manipulateAsync(
          photo.uri,
          [{ resize: { width: 800 } }],
          { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
        );
        
        setSelectedImage(resizedPhoto.uri);
        setShowCamera(false);
        analyzeImage(resizedPhoto.uri);
      } catch (error) {
        Alert.alert('Error', 'Failed to take picture');
      }
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        const resizedPhoto = await ImageManipulator.manipulateAsync(
          result.assets[0].uri,
          [{ resize: { width: 800 } }],
          { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
        );
        
        setSelectedImage(resizedPhoto.uri);
        analyzeImage(resizedPhoto.uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image');
    }
  };

  const analyzeImage = async (imageUri) => {
    if (!imageUri) return;

    setLoading(true);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'photo.jpg',
      });

      const response = await fetch(`${API_BASE_URL}/analyze-face`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        const errorData = await response.json();
        Alert.alert('Error', errorData.message || 'Failed to analyze image');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const renderResults = () => {
    if (!results || !results.success) return null;

    return (
      <ScrollView style={styles.resultsContainer}>
        <Text style={styles.resultsTitle}>Analysis Results</Text>
        
        {/* Show overlay image */}
        {results.overlay_image_base64 && (
          <View style={styles.overlayContainer}>
            <Text style={styles.sectionTitle}>Face Detection</Text>
            <Image
              source={{ uri: `data:image/png;base64,${results.overlay_image_base64}` }}
              style={styles.overlayImage}
              resizeMode="contain"
            />
          </View>
        )}

        {/* Show individual faces */}
        {results.faces.map((face, index) => (
          <View key={index} style={styles.faceResult}>
            <Text style={styles.faceTitle}>Face {face.face_id}</Text>
            
            <View style={styles.faceContent}>
              {/* Face crop */}
              <View style={styles.faceCropContainer}>
                <Image
                  source={{ uri: `data:image/png;base64,${face.face_crop_base64}` }}
                  style={styles.faceCrop}
                  resizeMode="contain"
                />
              </View>

              {/* Age predictions */}
              <View style={styles.ageContainer}>
                {face.harvard_age && (
                  <View style={styles.ageRow}>
                    <Text style={styles.ageLabel}>🎯 Harvard Age:</Text>
                    <Text style={styles.ageValue}>{face.harvard_age.toFixed(1)} years</Text>
                  </View>
                )}
                
                {face.deepface_age && (
                  <View style={styles.ageRow}>
                    <Text style={styles.ageLabel}>🤖 DeepFace Age:</Text>
                    <Text style={styles.ageValue}>{face.deepface_age.toFixed(1)} years</Text>
                  </View>
                )}

                <View style={styles.ageRow}>
                  <Text style={styles.ageLabel}>Detection:</Text>
                  <Text style={styles.ageValue}>{(face.confidence * 100).toFixed(1)}%</Text>
                </View>

                {/* Category indicator */}
                <View style={[styles.categoryBadge, styles[`category${face.category}`]]}>
                  <Text style={styles.categoryText}>
                    {face.category === 'young' ? '😊 Young' : 
                     face.category === 'adult' ? '😐 Adult' : 
                     face.category === 'senior' ? '👴 Senior' : '❓ Unknown'}
                  </Text>
                </View>

                {/* Warnings */}
                {face.warnings.map((warning, wIndex) => (
                  <Text key={wIndex} style={styles.warning}>⚠️ {warning}</Text>
                ))}
              </View>
            </View>
          </View>
        ))}
      </ScrollView>
    );
  };

  if (hasPermission === null) {
    return <View style={styles.container}><Text>Requesting camera permission...</Text></View>;
  }

  if (hasPermission === false) {
    return <View style={styles.container}><Text>No access to camera</Text></View>;
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="auto" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>📱 Face Age Estimator</Text>
        <Text style={styles.subtitle}>AI-powered age estimation</Text>
        
        {/* API Health Status */}
        <View style={styles.apiStatus}>
          <Text style={styles.apiStatusText}>
            API: {apiHealth?.status === 'healthy' ? '✅ Connected' : '❌ Offline'}
          </Text>
          <TouchableOpacity onPress={checkApiHealth} style={styles.refreshButton}>
            <Ionicons name="refresh" size={16} color="#666" />
          </TouchableOpacity>
        </View>
      </View>

      {/* Camera View */}
      {showCamera && (
        <View style={styles.cameraContainer}>
          <Camera
            style={styles.camera}
            type={Camera.Constants.Type.back}
            ref={setCameraRef}
          >
            <View style={styles.cameraOverlay}>
              <TouchableOpacity 
                style={styles.captureButton}
                onPress={takePicture}
              >
                <Ionicons name="camera" size={40} color="white" />
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.closeButton}
                onPress={() => setShowCamera(false)}
              >
                <Ionicons name="close" size={30} color="white" />
              </TouchableOpacity>
            </View>
          </Camera>
        </View>
      )}

      {/* Main Content */}
      {!showCamera && (
        <ScrollView style={styles.content}>
          {/* Action Buttons */}
          <View style={styles.buttonContainer}>
            <TouchableOpacity 
              style={[styles.button, styles.cameraButton]}
              onPress={() => setShowCamera(true)}
            >
              <Ionicons name="camera" size={24} color="white" />
              <Text style={styles.buttonText}>Take Photo</Text>
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.button, styles.galleryButton]}
              onPress={pickImage}
            >
              <Ionicons name="images" size={24} color="white" />
              <Text style={styles.buttonText}>Choose from Gallery</Text>
            </TouchableOpacity>
          </View>

          {/* Selected Image */}
          {selectedImage && (
            <View style={styles.imageContainer}>
              <Text style={styles.sectionTitle}>Selected Image</Text>
              <Image source={{ uri: selectedImage }} style={styles.selectedImage} />
            </View>
          )}

          {/* Loading Indicator */}
          {loading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
              <Text style={styles.loadingText}>Analyzing faces...</Text>
            </View>
          )}

          {/* Results */}
          {renderResults()}
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: 'white',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 10,
  },
  apiStatus: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  apiStatusText: {
    fontSize: 12,
    color: '#666',
    marginRight: 10,
  },
  refreshButton: {
    padding: 5,
  },
  content: {
    flex: 1,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 20,
    gap: 10,
  },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 10,
    gap: 8,
  },
  cameraButton: {
    backgroundColor: '#007AFF',
  },
  galleryButton: {
    backgroundColor: '#34C759',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 50,
  },
  captureButton: {
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 50,
    padding: 20,
    marginBottom: 20,
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    padding: 10,
  },
  imageContainer: {
    padding: 20,
    alignItems: 'center',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  selectedImage: {
    width: width - 40,
    height: 300,
    borderRadius: 10,
  },
  loadingContainer: {
    padding: 40,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  resultsContainer: {
    padding: 20,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  overlayContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  overlayImage: {
    width: width - 40,
    height: 300,
    borderRadius: 10,
  },
  faceResult: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  faceTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  faceContent: {
    flexDirection: 'row',
    gap: 15,
  },
  faceCropContainer: {
    flex: 1,
  },
  faceCrop: {
    width: 100,
    height: 100,
    borderRadius: 10,
  },
  ageContainer: {
    flex: 2,
  },
  ageRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  ageLabel: {
    fontSize: 14,
    color: '#666',
  },
  ageValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  categoryBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
    alignSelf: 'flex-start',
    marginTop: 5,
  },
  categoryyoung: {
    backgroundColor: '#E8F5E8',
  },
  categoryadult: {
    backgroundColor: '#E8F0FF',
  },
  categorysenior: {
    backgroundColor: '#FFF3E0',
  },
  categoryunknown: {
    backgroundColor: '#F5F5F5',
  },
  categoryText: {
    fontSize: 12,
    fontWeight: '600',
  },
  warning: {
    fontSize: 12,
    color: '#FF9500',
    marginTop: 5,
  },
}); 