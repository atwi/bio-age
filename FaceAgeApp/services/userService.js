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

// Save analysis result to user's history
export const saveAnalysisResult = async (userId, analysisData, imageBlob = null) => {
  try {
    const userRef = doc(db, 'users', userId);
    
    // Generate unique analysis ID
    const analysisId = `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    let imageUrl = null;
    
    // Upload image to Firebase Storage if provided
    if (imageBlob) {
      const imageRef = ref(storage, `user_photos/${userId}/${analysisId}.jpg`);
      const snapshot = await uploadBytes(imageRef, imageBlob);
      imageUrl = await getDownloadURL(snapshot.ref);
    }
    
    // Prepare analysis document
    const analysisDoc = {
      id: analysisId,
      timestamp: serverTimestamp(),
      imageUrl: imageUrl,
      results: {
        harvard: analysisData.age_harvard || null,
        deepface: analysisData.age_deepface || null,
        chatgpt: analysisData.age_chatgpt || null,
        confidence: analysisData.confidence || null
      },
      factors: analysisData.chatgpt_factors || null,
      faceCropBase64: analysisData.face_crop_base64 || null
    };
    
    // Add to user's analyses array and increment count
    await updateDoc(userRef, {
      analyses: arrayUnion(analysisDoc),
      analysisCount: increment(1),
      lastAnalysisAt: serverTimestamp()
    });
    
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
      .sort((a, b) => (b.timestamp?.seconds || 0) - (a.timestamp?.seconds || 0))
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