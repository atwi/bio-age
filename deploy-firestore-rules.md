# Deploy Firestore Security Rules

The HTTP 400 errors are caused by Firestore security rules blocking access. Follow these steps to fix it:

## Step 1: Access Firebase Console
1. Go to https://console.firebase.google.com/
2. Select your project: `trueage-b1941`
3. In the left sidebar, click on "Firestore Database"

## Step 2: Update Security Rules
1. Click on the "Rules" tab
2. Replace the existing rules with:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow users to read and write their own user document
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Allow authenticated users to read and write their own analyses
    match /users/{userId}/analyses/{analysisId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Deny all other access by default
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

## Step 3: Publish Rules
1. Click "Publish" to deploy the new rules
2. Wait for the rules to be deployed (usually takes a few seconds)

## Step 4: Test
1. Refresh your application
2. The HTTP 400 errors should stop
3. The GPT request counter should work properly

## Alternative: Temporary Permissive Rules
If you want to test quickly, you can temporarily use these rules (REMOVE IN PRODUCTION):

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if true;
    }
  }
}
```

**WARNING: The permissive rules allow anyone to read/write any data. Only use for testing!**
