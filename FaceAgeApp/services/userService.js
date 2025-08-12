import { 
  doc, 
  getDoc, 
  setDoc, 
  updateDoc, 
  arrayUnion, 
  increment,
  serverTimestamp 
} from 'firebase/firestore';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { db, storage } from '../firebase';

const sanitizeObject = (obj) => {
  if (!obj || typeof obj !== 'object') return obj;
  const out = {};
  Object.keys(obj).forEach((k) => {
    const v = obj[k];
    if (v === undefined) return; // drop undefined
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      const nested = sanitizeObject(v);
      if (nested !== undefined) out[k] = nested;
    } else {
      out[k] = v;
    }
  });
  return out;
};

// Save analysis result to user's history
export const saveAnalysisResult = async (userId, analysisData, imageBlob = null) => {
  if (!userId || !analysisData) {
    console.warn('saveAnalysisResult: Missing required parameters');
    return { success: false, error: 'Missing required parameters' };
  }

  try {
    const userRef = doc(db, 'users', userId);
    
    // Generate unique analysis ID
    const analysisId = `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    let imageUrl = null;
    let imagePath = null;
    
    // Upload image to Firebase Storage if provided and storage is available
    if (imageBlob && storage) {
      try {
        imagePath = `user_photos/${userId}/${analysisId}.jpg`;
        const imageRef = ref(storage, imagePath);
        const snapshot = await uploadBytes(imageRef, imageBlob);
        imageUrl = await getDownloadURL(snapshot.ref);
      } catch (uploadError) {
        console.warn('Non-fatal: failed to upload image:', uploadError?.message || uploadError);
        // Continue without image upload
      }
    }
    
    const timestampMs = Date.now();

    // Prepare a compact summary to keep user doc small
    const analysisSummary = {
      id: analysisId,
      timestampMs,
      imageUrl: imageUrl,
      results: {
        harvard: analysisData?.age_harvard ?? null,
        harvard_calibrated: analysisData?.age_harvard_calibrated ?? null,
        deepface: analysisData?.age_deepface ?? null,
        chatgpt: analysisData?.age_chatgpt ?? null,
        confidence: analysisData?.confidence ?? null,
      },
    };

    // Prepare detailed doc (no large base64 blobs)
    const analysisDetails = sanitizeObject({
      ...analysisSummary,
      factors: analysisData?.chatgpt_factors ?? null,
      // Store storage path instead of embedding large base64
      imagePath: imagePath,
      // Do NOT store faceCropBase64 to avoid exceeding Firestore doc limits
    });

    // Write detailed doc to subcollection
    const analysisRef = doc(db, 'users', userId, 'analyses', analysisId);
    const detailPayload = sanitizeObject({
      ...analysisDetails,
      createdAt: serverTimestamp(),
    });
    
    try {
      await setDoc(analysisRef, detailPayload);
    } catch (e) {
      console.warn('Non-fatal: failed to write analysis detail doc:', e?.message || e);
    }

    // Create/merge user doc with compact summary and counters (merge ensures doc is created if missing)
    const userPayload = sanitizeObject({
      analyses: arrayUnion(sanitizeObject(analysisSummary)),
      analysisCount: increment(1),
      lastAnalysisAt: serverTimestamp(),
      lastAnalysisId: analysisId,
    });
    
    try {
      await setDoc(
        userRef,
        userPayload,
        { merge: true }
      );
    } catch (e) {
      console.warn('Non-fatal: failed to write user summary doc:', e?.message || e);
    }
    
    return { success: true, analysisId };
  } catch (error) {
    console.error('Error saving analysis result:', error);
    return { success: false, error: error.message };
  }
};

// Get user's analysis history
export const getUserAnalysisHistory = async (userId, limit = 20) => {
  try {
    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);
    
    if (!userSnap.exists()) {
      return { success: false, error: 'User not found' };
    }
    
    const userData = userSnap.data();
    const analyses = userData.analyses || [];
    
    // Sort by timestamp (newest first) and limit results
    const sortedAnalyses = analyses
      .sort((a, b) => (b.timestamp?.seconds || b.timestampMs || 0) - (a.timestamp?.seconds || a.timestampMs || 0))
      .slice(0, limit);
    
    return { success: true, analyses: sortedAnalyses };
  } catch (error) {
    console.error('Error getting user analysis history:', error);
    return { success: false, error: error.message };
  }
};

// Get user profile data
export const getUserProfile = async (userId) => {
  try {
    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);
    
    if (!userSnap.exists()) {
      return { success: false, error: 'User not found' };
    }
    
    return { success: true, profile: userSnap.data() };
  } catch (error) {
    console.error('Error getting user profile:', error);
    return { success: false, error: error.message };
  }
};

// Update user profile
export const updateUserProfile = async (userId, updates) => {
  try {
    const userRef = doc(db, 'users', userId);
    
    await updateDoc(userRef, {
      ...updates,
      updatedAt: serverTimestamp()
    });
    
    return { success: true };
  } catch (error) {
    console.error('Error updating user profile:', error);
    return { success: false, error: error.message };
  }
};

// Delete analysis from user's history
export const deleteAnalysis = async (userId, analysisId) => {
  try {
    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);
    
    if (!userSnap.exists()) {
      return { success: false, error: 'User not found' };
    }
    
    const userData = userSnap.data();
    const analyses = userData.analyses || [];
    
    // Filter out the analysis to delete
    const updatedAnalyses = analyses.filter(analysis => analysis.id !== analysisId);
    
    await updateDoc(userRef, {
      analyses: updatedAnalyses,
      analysisCount: increment(-1)
    });
    
    return { success: true };
  } catch (error) {
    console.error('Error deleting analysis:', error);
    return { success: false, error: error.message };
  }
}; 