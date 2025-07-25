import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, OAuthProvider } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';

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
const app = initializeApp(firebaseConfig);

// Initialize Firebase services
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);

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

export default app; 