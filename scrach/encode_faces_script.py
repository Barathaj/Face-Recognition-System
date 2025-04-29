import os
import cv2
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Initialize Firebase
cred = credentials.Certificate("service.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://bigvision-22c68-default-rtdb.firebaseio.com/",
    'storageBucket': "bigvision-22c68.firebasestorage.app"
})

# Initialize face detector
face_detector = MTCNN()

# Load FaceNet model
facenet_model_path = 'models/facenet_keras.h5'
if not os.path.exists(facenet_model_path):
    print(f"ERROR: FaceNet model not found at {facenet_model_path}")
    print("Please download the model from: https://github.com/nyoki-mtl/keras-facenet")
    exit(1)

facenet_model = load_model(facenet_model_path)
print("FaceNet model loaded successfully")

# Get the database reference
ref = db.reference('Students')

# Function to preprocess face for FaceNet
def preprocess_face(face_img, required_size=(160, 160)):
    """Preprocess face image for FaceNet input"""
    face_img = cv2.resize(face_img, required_size)
    face_img = face_img.astype('float32')
    # Standardize pixel values
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    return face_img

# Function to get embedding from FaceNet
def get_embedding(face_img):
    """Get embedding vector from FaceNet model"""
    # Expand dimensions to match the model's expected input
    samples = np.expand_dims(face_img, axis=0)
    # Get embeddings
    yhat = facenet_model.predict(samples)
    return yhat[0]

# Function to detect and preprocess face from image
def extract_face(img_path):
    """Extract face from image and return preprocessed face"""
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return None
    
    # Convert to RGB for MTCNN
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detector.detect_faces(rgb_img)
    
    if not results:
        print(f"No face detected in {img_path}")
        return None
    
    # Get largest face (assuming this is the main person in the image)
    largest_area = 0
    largest_face = None
    
    for result in results:
        x, y, w, h = result['box']
        # Handle negative values sometimes returned by MTCNN
        x, y = max(0, x), max(0, y)
        area = w * h
        
        if area > largest_area:
            largest_area = area
            face_img = rgb_img[y:y+h, x:x+w]
            largest_face = face_img
    
    if largest_face is None:
        print(f"Could not process face in {img_path}")
        return None
    
    # Convert back to BGR for consistency with OpenCV
    largest_face = cv2.cvtColor(largest_face, cv2.COLOR_RGB2BGR)
    # Preprocess for FaceNet
    processed_face = preprocess_face(largest_face)
    
    return processed_face

print("Encoding Faces...")

# Initialize lists to store encodings and IDs
encodeListKnown = []
studentIds = []

# Get all student data
students = ref.get()

for key, value in students.items():
    try:
        # Get student ID
        student_id = key
        
        # Download student image from Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(f'images/{student_id}.jpeg')
        
        if not blob.exists():
            print(f"Image for student {student_id} does not exist in storage. Skipping.")
            continue
        
        # Create a temporary file path
        temp_img_path = f"temp_{student_id}.jpeg"
        blob.download_to_filename(temp_img_path)
        
        # Extract face from image
        processed_face = extract_face(temp_img_path)
        
        # Clean up temporary file
        os.remove(temp_img_path)
        
        if processed_face is not None:
            # Get embedding
            face_encoding = get_embedding(processed_face)
            
            # Add to lists
            encodeListKnown.append(face_encoding)
            studentIds.append(student_id)
            
            print(f"Encoded student {student_id}: {value['name']}")
        else:
            print(f"Failed to encode student {student_id}: {value['name']}")
    
    except Exception as e:
        print(f"Error processing student {key}: {e}")

print(f"Encoding Complete! {len(encodeListKnown)} faces encoded.")

# Save encodings to file
print("Saving encodings to file...")
file = open("FaceNetEncodeFile.p", "wb")
pickle.dump([encodeListKnown, studentIds], file)
file.close()
print("File saved successfully!")