// Use modular Firebase imports for better tree shaking
import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, OAuthProvider } from 'firebase/auth';
import { getFirestore, initializeFirestore, connectFirestoreEmulator, setLogLevel } from 'firebase/firestore';
import { getStorage, connectStorageEmulator } from 'firebase/storage';

// Your Firebase config - Replace with your actual config from Firebase Console
const firebaseConfig = {      
    apiKey: "AIzaSyDNMeWyOpbPMqKTKL333CBgraW3FAoPyFs",
    authDomain: "trueage-b1941.firebaseapp.com",
    projectId: "trueage-b1941",
    storageBucket: "trueage-b1941.firebasestorage.app",
    messagingSenderId: "465255398940",
    appId: "1:465255398940:web:79eb06edb116e960cd69d9"
};

// Initialize Firebase
let app;
try {
  app = initializeApp(firebaseConfig);
} catch (error) {
  console.error('Firebase initialization error:', error);
  // Create a minimal app for basic functionality
  app = initializeApp({
    apiKey: firebaseConfig.apiKey,
    authDomain: firebaseConfig.authDomain,
    projectId: firebaseConfig.projectId,
  });
}

// Initialize Firebase services
export const auth = getAuth(app);

let db;
try {
  db = initializeFirestore(app, {
    ignoreUndefinedProperties: true,
    // Use more robust connection settings to avoid Write/channel stream issues
    experimentalForceLongPolling: true,
    useFetchStreams: false,
    // Add connection timeout and retry settings
    cacheSizeBytes: 50 * 1024 * 1024, // 50MB cache
  });
} catch (error) {
  console.error('Firestore initialization error:', error);
  // Fallback to basic Firestore initialization
  db = getFirestore(app);
}
export { db };

let storage;
try {
  storage = getStorage(app);
} catch (error) {
  console.error('Firebase Storage initialization error:', error);
  storage = null;
}
export { storage };

// Auth providers
export const googleProvider = new GoogleAuthProvider();
export const appleProvider = new OAuthProvider('apple.com');

// Configure providers
googleProvider.setCustomParameters({
  prompt: 'select_account'
});

appleProvider.setCustomParameters({
  locale: 'en'
});

// Reduce console noise from Firestore transport retries
try { setLogLevel('error'); } catch {}

// Add error handling for Firestore connection issues
if (typeof window !== 'undefined') {
  // Handle potential Firestore connection errors in web environment
  window.addEventListener('error', (event) => {
    if (event.error && event.error.message && event.error.message.includes('Firestore')) {
      console.warn('Firestore connection issue detected, this is usually non-critical');
    }
  });
}

export default app; 