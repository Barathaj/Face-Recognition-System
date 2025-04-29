import os
import pickle
import numpy as np
import cv2
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image

# Initialize Firebase
cred = credentials.Certificate("service.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://bigvision-22c68-default-rtdb.firebaseio.com/",
    'storageBucket': "bigvision-22c68.firebasestorage.app"
})

bucket = storage.bucket()

# Initialize video capture
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Load background image
imgBackground = cv2.imread('resources/background2.png')

# Importing the mode images into a list
folderModePath = 'resources/new_modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Initialize face detection model (MTCNN)
face_detector = MTCNN()

# Load FaceNet model
# You need to download the FaceNet Keras model first
# Available at: https://github.com/nyoki-mtl/keras-facenet
facenet_model_path = 'models/facenet_keras.h5'
if not os.path.exists(facenet_model_path):
    print(f"ERROR: FaceNet model not found at {facenet_model_path}")
    exit(1)

facenet_model = load_model(facenet_model_path)
print("FaceNet model loaded successfully")

# Load the encoding file (or create a new one if not exists)
print("Loading Encode File...")
encode_file_path = 'FaceNetEncodeFile.p'
if os.path.exists(encode_file_path):
    file = open(encode_file_path, 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, studentIds = encodeListKnownWithIds
    print("Encode File Loaded")
else:
    print("No existing encode file found. Will create when needed.")
    encodeListKnown = []
    studentIds = []

# Constants for timing (in frames, assuming ~25fps)
ACTIVE_DISPLAY_TIME = 5 * 25      # 5 seconds for "ACTIVE" screen
DETAILS_DISPLAY_TIME = 5 * 25     # 5 seconds for student details screen
MARKED_DISPLAY_TIME = 2 * 25      # 2 seconds for "MARKED" screen (green checkmark)
ALREADY_MARKED_DISPLAY_TIME = 3 * 25  # 3 seconds for "ALREADY MARKED" screen
COOLDOWN_PERIOD = 60              # 1 minute cooldown before allowing re-marking
SIMILARITY_THRESHOLD = 0.85       # Threshold for face match confidence (higher = stricter)

# Mode Types:
# 0: Scanning/Active mode
# 1: Student details mode (photo, ID, name)
# 2: Marked mode (green checkmark)
# 3: Already marked mode

modeType = 0
counter = 0
id = -1
imgStudent = None
studentInfo = None
marked_students = {}  # To track recently marked students {id: timestamp}
face_detected = False

def preprocess_face(face_img, required_size=(160, 160)):
    """Preprocess face image for FaceNet input"""
    face_img = cv2.resize(face_img, required_size)
    face_img = face_img.astype('float32')
    # Standardize pixel values
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    return face_img

def get_embedding(face_img):
    """Get embedding vector from FaceNet model"""
    # Expand dimensions to match the model's expected input
    samples = np.expand_dims(face_img, axis=0)
    # Get embeddings
    yhat = facenet_model.predict(samples)
    return yhat[0]

def detect_faces(img):
    """Detect faces using MTCNN and return face locations and aligned faces"""
    # Convert to RGB for MTCNN
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect faces
    results = face_detector.detect_faces(rgb_img)
    
    face_locations = []
    aligned_faces = []
    
    for result in results:
        # Get face box coordinates
        x, y, w, h = result['box']
        # Handle negative values sometimes returned by MTCNN
        x, y = max(0, x), max(0, y)
        face_img = rgb_img[y:y+h, x:x+w]
        
        # Skip if face is too small
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            continue
            
        # Process and align face
        try:
            # Convert back to BGR for consistency with OpenCV
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            # Preprocess for FaceNet
            processed_face = preprocess_face(face_img)
            # Add to results
            face_locations.append((y, x+w, y+h, x))  # Format: (top, right, bottom, left)
            aligned_faces.append(processed_face)
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
            
    return face_locations, aligned_faces

def compare_faces(known_embeddings, face_embedding, threshold=SIMILARITY_THRESHOLD):
    """Compare a face embedding with a list of known embeddings"""
    if len(known_embeddings) == 0:
        return [], []
    
    # Calculate similarity for each known face
    similarities = []
    for emb in known_embeddings:
        similarity = cosine_similarity([emb], [face_embedding])[0][0]
        similarities.append(similarity)
    
    # Create match results
    matches = [sim >= threshold for sim in similarities]
    
    return matches, similarities

# Draw centered text function
def draw_centered_text(image, text, box_x, box_y, box_w, box_h, font_scale=0.6, thickness=2):
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x = box_x + (box_w - text_width) // 2
    y = box_y + (box_h + text_height) // 2 - 5  # fine-tuned for vertical alignment
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness)  # black color

print("Starting face attendance system with FaceNet...")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to get frame from camera")
        continue

    # Create a copy of the background for each frame
    imgBackground = cv2.imread('resources/background2.png')
    
    # Resize webcam feed to match the green box perfectly
    imgResized = cv2.resize(img, (655, 380))

    # Define top-left coordinate of the green box
    x, y = 137, 244

    # Paste webcam feed onto background
    imgBackground[y:y+380, x:x+655] = imgResized
    
    # Create a copy to work with for drawing
    imgBackgroundCopy = imgBackground.copy()

    # Update the right panel image based on current mode
    panelResized = cv2.resize(imgModeList[modeType], (438, 714))
    panel_x, panel_y = 899, 23
    imgBackgroundCopy[panel_y:panel_y+714, panel_x:panel_x+438] = panelResized

    # Process face detection only if we're in scanning mode
    if modeType == 0:
        # Detect and align faces
        face_locations, processed_faces = detect_faces(img)
        
        # Reset face_detected flag
        face_detected = False
        
        if face_locations and processed_faces:
            for face_loc, processed_face in zip(face_locations, processed_faces):
                # Get the embedding for the current face
                face_embedding = get_embedding(processed_face)
                
                # Compare with known faces
                matches, similarities = compare_faces(encodeListKnown, face_embedding)
                
                if any(matches):
                    # Find the best match
                    match_index = np.argmax(similarities)
                    
                    # If match confidence is good enough
                    if matches[match_index]:
                        # Draw rectangle around face
                        y1, x2, y2, x1 = face_loc
                        bbox = x + x1, y + y1, x2 - x1, y2 - y1
                        imgBackgroundCopy = cvzone.cornerRect(imgBackgroundCopy, bbox, rt=0)
                        
                        # Get student ID
                        id = studentIds[match_index]
                        face_detected = True
                        
                        # Check if this student was recently marked (within cooldown period)
                        current_time = datetime.now()
                        if id in marked_students:
                            time_diff = (current_time - marked_students[id]).total_seconds()
                            
                            # If within cooldown period, show "Already Marked" screen
                            if time_diff < COOLDOWN_PERIOD:
                                counter = 1
                                modeType = 3  # Switch to "Already Marked" mode
                                
                                # Update right panel immediately for "Already Marked"
                                panelResized = cv2.resize(imgModeList[modeType], (438, 714))
                                imgBackgroundCopy[panel_y:panel_y+714, panel_x:panel_x+438] = panelResized
                                
                                print(f"Student ID {id} already marked - showing 'Already Marked' screen")
                                continue
                        
                        # Fetch student data
                        try:
                            studentInfo = db.reference(f'Students/{id}').get()
                            print(f"Retrieved student info for ID {id}")
                            
                            # Get student image from storage
                            try:
                                blob = bucket.get_blob(f'images/{id}.jpeg')
                                if blob:
                                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                                    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                                    print(f"Successfully loaded student image for ID {id}")
                                else:
                                    print(f"No image found for student ID {id}")
                                    imgStudent = None
                            except Exception as e:
                                print(f"Error loading student image: {e}")
                                imgStudent = None
                            
                            # Update attendance in database
                            if studentInfo and 'last_attendance_time' in studentInfo:
                                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], 
                                                                "%Y-%m-%d %H:%M:%S")
                                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                                
                                # Update attendance if enough time has passed since last mark
                                if secondsElapsed > 30:  # Database cooldown check
                                    ref = db.reference(f'Students/{id}')
                                    studentInfo['total_attendance'] += 1
                                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    
                                    # Record this student as recently marked
                                    marked_students[id] = datetime.now()
                                    print(f"Updated attendance for student ID {id}")
                                    
                                    # Switch to "Student Details" mode (photo, ID, name)
                                    counter = 1
                                    modeType = 1
                                    
                                    # Update right panel immediately for "Student Details"
                                    panelResized = cv2.resize(imgModeList[modeType], (438, 714))
                                    imgBackgroundCopy[panel_y:panel_y+714, panel_x:panel_x+438] = panelResized
                                else:
                                    # Student was marked too recently (database perspective)
                                    counter = 1
                                    modeType = 3  # Show as "Already Marked"
                                    
                                    # Update right panel immediately
                                    panelResized = cv2.resize(imgModeList[modeType], (438, 714))
                                    imgBackgroundCopy[panel_y:panel_y+714, panel_x:panel_x+438] = panelResized
                            else:
                                print(f"No attendance data for student ID {id}")
                        except Exception as e:
                            print(f"Error processing student data: {e}")
                            counter = 0
                            modeType = 0
    
    # Handle mode transitions based on counters
    if counter > 0:
        counter += 1
        
        # After showing student details screen for specified time, transition to marked screen
        if modeType == 1 and counter >= DETAILS_DISPLAY_TIME:
            print("Transitioning from details to marked mode")
            modeType = 2  # Switch to marked mode (green checkmark)
            counter = 1  # Reset counter for marked mode
            
            # Update the right panel immediately for marked
            panelResized = cv2.resize(imgModeList[modeType], (438, 714))
            imgBackgroundCopy[panel_y:panel_y+714, panel_x:panel_x+438] = panelResized
        
        # When in student details mode, display student information
        if modeType == 1:
            if studentInfo:
                # Display student information
                try:
                    # Draw ID
                    if id is not None:
                        draw_centered_text(imgBackgroundCopy, str(id), 1025, 555, 100, -170)

                    # Draw Name
                    if 'name' in studentInfo:
                        draw_centered_text(imgBackgroundCopy, studentInfo['name'], 1025, 555, 200, 40)

                    # Display student image
                    if imgStudent is not None and isinstance(imgStudent, np.ndarray) and imgStudent.size > 0:
                        try:
                            imgStudentResized = cv2.resize(imgStudent, (216, 216))
                            imgBackgroundCopy[150:150 + 216, 1009:1009 + 216] = imgStudentResized
                        except Exception as e:
                            print(f"Error displaying student image: {e}")
                except Exception as e:
                    print(f"Error displaying student info: {e}")
        
        # After showing "Already Marked" for specified time, return to scanning mode
        elif modeType == 3 and counter >= ALREADY_MARKED_DISPLAY_TIME:
            counter = 0
            modeType = 0
            studentInfo = None
            imgStudent = None
        
        # After showing "Marked" screen (checkmark) for specified time, return to scanning mode
        elif modeType == 2 and counter >= MARKED_DISPLAY_TIME:
            counter = 0
            modeType = 0
            studentInfo = None
            imgStudent = None
    
    # Handle cleanup of marked_students dictionary (remove entries older than cooldown period)
    current_time = datetime.now()
    expired_ids = []
    for student_id, mark_time in marked_students.items():
        if (current_time - mark_time).total_seconds() > COOLDOWN_PERIOD:
            expired_ids.append(student_id)
    
    for student_id in expired_ids:
        del marked_students[student_id]
    
    # Display the final image
    cv2.imshow("Face Attendance", imgBackgroundCopy)
    
    # Check for key press (ESC to exit)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()