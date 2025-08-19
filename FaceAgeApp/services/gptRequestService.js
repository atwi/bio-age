import AsyncStorage from '@react-native-async-storage/async-storage';
import { doc, getDoc, updateDoc, increment, setDoc, serverTimestamp } from 'firebase/firestore';
import { db, auth } from '../firebase';

const GPT_REQUESTS_KEY = 'gpt_requests_used';
const ANONYMOUS_LIMIT = 2;
const AUTHENTICATED_LIMIT = 10;

// Get GPT request limits based on authentication status
export const getGptRequestLimits = (isAuthenticated) => {
  return {
    limit: isAuthenticated ? AUTHENTICATED_LIMIT : ANONYMOUS_LIMIT,
    used: 0, // Will be populated by getGptRequestsUsed
    remaining: isAuthenticated ? AUTHENTICATED_LIMIT : ANONYMOUS_LIMIT
  };
};

// Get current GPT requests used (for anonymous users)
export const getAnonymousGptRequestsUsed = async () => {
  try {
    const used = await AsyncStorage.getItem(GPT_REQUESTS_KEY);
    return used ? parseInt(used, 10) : 0;
  } catch (error) {
    console.error('Error getting anonymous GPT requests:', error);
    return 0;
  }
};

// Increment anonymous GPT requests used
export const incrementAnonymousGptRequests = async () => {
  try {
    const currentUsed = await getAnonymousGptRequestsUsed();
    const newUsed = currentUsed + 1;
    await AsyncStorage.setItem(GPT_REQUESTS_KEY, newUsed.toString());
    return newUsed;
  } catch (error) {
    console.error('Error incrementing anonymous GPT requests:', error);
    return 0;
  }
};

// Get authenticated user's GPT requests used
export const getAuthenticatedGptRequestsUsed = async (userId) => {
  if (!userId) return 0;
  
  // Verify user is authenticated
  const currentUser = auth.currentUser;
  console.log('[GPT] Current auth user:', currentUser?.uid);
  console.log('[GPT] Requested user ID:', userId);
  if (!currentUser || currentUser.uid !== userId) {
    console.error('[GPT] User authentication mismatch or not authenticated');
    console.error('[GPT] Current user:', currentUser?.uid, 'Requested:', userId);
    return 0;
  }
  
  try {
    console.log('[GPT] Getting authenticated GPT requests for user:', userId);
    console.log('[GPT] Auth current user:', auth.currentUser?.uid);
    console.log('[GPT] Auth current user email:', auth.currentUser?.email);
    console.log('[GPT] Auth current user displayName:', auth.currentUser?.displayName);
    
    // Get the current user's ID token to verify authentication
    try {
      const idToken = await auth.currentUser?.getIdToken();
      console.log('[GPT] ID token obtained:', idToken ? 'Yes' : 'No');
    } catch (tokenError) {
      console.error('[GPT] Error getting ID token:', tokenError);
    }
    
    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);
    
    if (userSnap.exists()) {
      const userData = userSnap.data();
      console.log('[GPT] User data retrieved:', userData);
      // Initialize gptRequestsUsed if it doesn't exist (for existing users)
      if (userData.gptRequestsUsed === undefined) {
        console.log('[GPT] Initializing gptRequestsUsed for existing user:', userId);
        try {
          await updateDoc(userRef, {
            gptRequestsUsed: 0
          });
          console.log('[GPT] Successfully initialized gptRequestsUsed');
        } catch (updateError) {
          console.error('[GPT] Error updating user document:', updateError);
          // If we can't update, return 0 as fallback
          return 0;
        }
        return 0;
      }
      const used = userData.gptRequestsUsed || 0;
      console.log('[GPT] Current GPT requests used:', used);
      return used;
    }
    
    // User document doesn't exist, create it
    console.log('[GPT] User document does not exist, creating it...');
    const userData = {
      uid: userId,
      email: currentUser.email || null,
      displayName: currentUser.displayName || null,
      photoURL: currentUser.photoURL || null,
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
    
    try {
      await setDoc(userRef, userData);
      console.log('[GPT] Successfully created user document');
      return 0; // New user starts with 0 GPT requests used
    } catch (createError) {
      console.error('[GPT] Error creating user document:', createError);
      // If we can't create, return 0 as fallback
      return 0;
    }
  } catch (error) {
    console.error('[GPT] Error getting authenticated GPT requests:', error);
    console.error('[GPT] Error details:', error.code, error.message);
    return 0;
  }
};

// Increment authenticated user's GPT requests used
export const incrementAuthenticatedGptRequests = async (userId) => {
  if (!userId) return 0;
  
  // Verify user is authenticated
  const currentUser = auth.currentUser;
  console.log('[GPT] Current auth user for increment:', currentUser?.uid);
  console.log('[GPT] Requested user ID for increment:', userId);
  if (!currentUser || currentUser.uid !== userId) {
    console.error('[GPT] User authentication mismatch or not authenticated for increment');
    return 0;
  }
  
  try {
    console.log('[GPT] Incrementing authenticated GPT requests for user:', userId);
    const userRef = doc(db, 'users', userId);
    await updateDoc(userRef, {
      gptRequestsUsed: increment(1)
    });
    
    // Get updated count
    const userSnap = await getDoc(userRef);
    const updatedCount = userSnap.data().gptRequestsUsed || 0;
    console.log('[GPT] Successfully incremented GPT requests. New count:', updatedCount);
    return updatedCount;
  } catch (error) {
    console.error('[GPT] Error incrementing authenticated GPT requests:', error);
    console.error('[GPT] Error details:', error.code, error.message);
    return 0;
  }
};

// Get current GPT request status
export const getGptRequestStatus = async (user) => {
  const isAuthenticated = !!user;
  const limits = getGptRequestLimits(isAuthenticated);
  
  try {
    console.log('[GPT] Getting request status for user:', user?.uid || 'anonymous');
    console.log('[GPT] Is authenticated:', isAuthenticated);
    console.log('[GPT] Current auth user:', auth.currentUser?.uid);
    console.log('[GPT] Auth state:', auth.currentUser ? 'authenticated' : 'not authenticated');
    
    if (isAuthenticated) {
      limits.used = await getAuthenticatedGptRequestsUsed(user.uid);
    } else {
      limits.used = await getAnonymousGptRequestsUsed();
    }
    
    limits.remaining = Math.max(0, limits.limit - limits.used);
    
    const status = {
      ...limits,
      canUseGpt: limits.remaining > 0,
      isAuthenticated
    };
    
    console.log('[GPT] Status result:', status);
    return status;
  } catch (error) {
    console.error('[GPT] Error getting request status:', error);
    console.error('[GPT] Error details:', error.code, error.message);
    // Fallback: assume user can use GPT if we can't determine their status
    return {
      ...limits,
      used: 0,
      remaining: limits.limit,
      canUseGpt: true,
      isAuthenticated,
      error: true
    };
  }
};

// Check if user can use GPT and increment if they can
export const checkAndIncrementGptRequests = async (user) => {
  try {
    const status = await getGptRequestStatus(user);
    
    if (!status.canUseGpt) {
      return { success: false, status };
    }
    
    // Increment the counter
    if (status.isAuthenticated) {
      await incrementAuthenticatedGptRequests(user.uid);
    } else {
      await incrementAnonymousGptRequests();
    }
    
    // Get updated status
    const updatedStatus = await getGptRequestStatus(user);
    
    return { success: true, status: updatedStatus };
  } catch (error) {
    console.error('[GPT] Error checking/incrementing requests:', error);
    // Fallback: allow the request to proceed if we can't track it
    return { success: true, status: { error: true, canUseGpt: true } };
  }
};

// Reset anonymous user's GPT requests (for testing)
export const resetAnonymousGptRequests = async () => {
  try {
    await AsyncStorage.removeItem(GPT_REQUESTS_KEY);
    return true;
  } catch (error) {
    console.error('Error resetting anonymous GPT requests:', error);
    return false;
  }
};
