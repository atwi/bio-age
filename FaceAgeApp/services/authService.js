import { 
  signInWithPopup, 
  signOut as firebaseSignOut,
  onAuthStateChanged,
  sendSignInLinkToEmail,
  isSignInWithEmailLink,
  signInWithEmailLink,
} from 'firebase/auth';
import { doc, setDoc, getDoc, serverTimestamp } from 'firebase/firestore';
import { auth, googleProvider, appleProvider, db } from '../firebase';

// Storage key for pending email link sign-in
const EMAIL_STORAGE_KEY = 'trueageEmailForSignIn';

// Sign in with Google
export const signInWithGoogle = async () => {
  try {
    const result = await signInWithPopup(auth, googleProvider);
    const user = result.user;
    
    // Create or update user document in Firestore
    await createOrUpdateUserDocument(user);
    
    return { success: true, user };
  } catch (error) {
    console.error('Google sign in error:', error);
    return { success: false, error: error.message };
  }
};

// Sign in with Apple
export const signInWithApple = async () => {
  try {
    const result = await signInWithPopup(auth, appleProvider);
    const user = result.user;
    
    // Create or update user document in Firestore
    await createOrUpdateUserDocument(user);
    
    return { success: true, user };
  } catch (error) {
    console.error('Apple sign in error:', error);
    return { success: false, error: error.message };
  }
};

// Send passwordless sign-in link to email (web)
export const sendMagicLink = async (email) => {
  try {
    if (!email || typeof email !== 'string') {
      throw new Error('Please enter a valid email');
    }
    const actionCodeSettings = {
      url: (typeof window !== 'undefined' ? window.location.origin : 'https://trueage.app'),
      handleCodeInApp: true,
    };
    await sendSignInLinkToEmail(auth, email, actionCodeSettings);
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(EMAIL_STORAGE_KEY, email);
    }
    return { success: true };
  } catch (error) {
    console.error('sendMagicLink error:', error);
    return { success: false, error: error.message };
  }
};

export const isMagicLink = (url) => {
  try {
    return isSignInWithEmailLink(auth, url);
  } catch (e) {
    return false;
  }
};

export const completeMagicLinkSignIn = async (url, emailFromUser) => {
  try {
    let email = emailFromUser;
    if (!email && typeof window !== 'undefined' && window.localStorage) {
      email = window.localStorage.getItem(EMAIL_STORAGE_KEY) || '';
    }
    if (!email) {
      throw new Error('Email required to complete sign-in');
    }
    const result = await signInWithEmailLink(auth, email, url);
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(EMAIL_STORAGE_KEY);
    }
    const user = result.user;
    await createOrUpdateUserDocument(user);
    return { success: true, user };
  } catch (error) {
    console.error('completeMagicLinkSignIn error:', error);
    return { success: false, error: error.message };
  }
};

// Sign out
export const signOut = async () => {
  try {
    await firebaseSignOut(auth);
    return { success: true };
  } catch (error) {
    console.error('Sign out error:', error);
    return { success: false, error: error.message };
  }
};

// Create or update user document in Firestore
const createOrUpdateUserDocument = async (user) => {
  if (!user || !user.uid) return;
  
  try {
    const userRef = doc(db, 'users', user.uid);
    const userSnap = await getDoc(userRef);
    
    if (!userSnap.exists()) {
      // Create new user document
      const userData = {
        uid: user.uid,
        email: user.email || null,
        displayName: user.displayName || null,
        photoURL: user.photoURL || null,
        createdAt: serverTimestamp(),
        lastLoginAt: serverTimestamp(),
        subscription: 'free',
        analysisCount: 0,
        analyses: [],
        gptRequestsUsed: 0
      };
      
      // Remove undefined values to prevent Firestore errors
      Object.keys(userData).forEach(key => {
        if (userData[key] === undefined) {
          delete userData[key];
        }
      });
      
      await setDoc(userRef, userData);
    } else {
      // Update last login time
      await setDoc(userRef, {
        lastLoginAt: serverTimestamp()
      }, { merge: true });
    }
  } catch (error) {
    console.error('Error creating/updating user document:', error);
    // Don't throw - this is non-critical for app functionality
  }
};

// Listen to auth state changes
export const onAuthStateChange = (callback) => {
  return onAuthStateChanged(auth, callback);
};

// Get current user
export const getCurrentUser = () => {
  return auth.currentUser;
}; 