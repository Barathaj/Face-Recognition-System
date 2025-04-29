import cv2
import face_recognition
import pickle
import os
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import datetime
bg_img = cv2.imread("E:\cv\cv_lib\resources\register_bg.png")
x1, y1 = 363, 132
x2, y2 = 1002, 568
frame_width = x2 - x1
frame_height = y2 - y1
# Initialize Firebase
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("E:\cv\cv_lib\service2.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://bigvision-22c68-default-rtdb.firebaseio.com/",
            'storageBucket': "bigvision-22c68.firebasestorage.app"
        })
    
    return db.reference('Students')

# Create images folder if it doesn't exist
def setup_directories():
    if not os.path.exists('E:\cv\cv_lib\images'):
        os.makedirs('E:\cv\cv_lib\images')
    print("Directory 'E:\\cv\\cv_lib\\images' is ready")

# Capture image from webcam
def capture_face():
    bg_img = cv2.imread("register_bg.png")
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    print("Please look at the camera...")
    print("Press 'c' to capture or 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        output_img = bg_img.copy()
        output_img[y1:y2, x1:x2] = resized_frame
        # Display countdown timer if space is pressed
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        
        # Draw rectangle around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Registration System', output_img)
        
        # Keyboard input
        key = cv2.waitKey(1)
        print(f"Key pressed: {key}")  # Debug line
        # If 'c' is pressed, capture the image
        if key == ord('c'):
            # Check if a face is detected
            if len(face_locations) == 0:
                print("No face detected. Please try again.")
                continue
            
            if len(face_locations) > 1:
                print("Multiple faces detected. Please ensure only one person is in the frame.")
                continue
            
            # Capture successful
            print("Face captured successfully!")
            cap.release()
            cv2.destroyAllWindows()
            return frame
        
        # If 'q' is pressed, quit
        elif key == ord('q'):
            print("Capture cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    return None

# Get student details
def get_student_details():
    student_id = input("Enter student ID: ")
    name = input("Enter student name: ")
    major = input("Enter student major: ")
    year = int(input("Enter current year of study: "))
    
    return {
        "id": student_id,
        "name": name,
        "major": major,
        "year": year,
        "starting_year": datetime.datetime.now().year - year + 1,
        "total_attendance": 0,
        "standing": "G",
        "last_attendance_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Save image to file
def save_image(frame, student_id):
    if frame is None:
        return False
    
    filename = f"E:\cv\cv_lib\images\{student_id}.jpeg"
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")
    return True

# Upload image to Firebase Storage
def upload_to_storage(student_id):
    local_path = f"E:\\cv\\cv_lib\\images\\{student_id}.jpeg"
    firebase_path = f"images/{student_id}.jpeg"  # Fixed path in Firebase

    if not os.path.exists(local_path):
        print(f"Error: File {local_path} does not exist")
        return False

    bucket = storage.bucket()
    blob = bucket.blob(firebase_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to Firebase as {firebase_path}")
    return True

# Store student data in Firebase Realtime Database
def store_in_database(ref, student_data):
    student_id = student_data["id"]
    
    # Remove ID from data to be stored
    data_to_store = student_data.copy()
    data_to_store.pop("id")
    
    # Store in Firebase
    ref.child(student_id).set(data_to_store)
    print(f"Stored student data in Firebase Database")
    return True

# Update encodings file
def update_encodings():
    print("Updating encodings...")
    
    # Get all images
    folderPath = 'E:\cv\cv_lib\images'
    pathList = os.listdir(folderPath)
    imgList = []
    studentIds = []
    
    for path in pathList:
        # Read image
        img = cv2.imread(os.path.join(folderPath, path))
        if img is None:
            print(f"Error: Could not read {path}")
            continue
        
        imgList.append(img)
        studentIds.append(os.path.splitext(path)[0])
    
    # Generate encodings
    encodeListKnown = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeListKnown.append(encode)
        except IndexError:
            print(f"No face found in one of the images. Skipping.")
            continue
    
    # Save encodings to file
    encodeListKnownWithIds = [encodeListKnown, studentIds]
    
    with open("E:\cv\cv_lib\EncodeFile.p", 'wb') as file:
        pickle.dump(encodeListKnownWithIds, file)
    
    print("Encodings updated and saved to file")
    return True

# Main function
def main():
    print("==== Face Registration System ====")
    
    # Setup
    setup_directories()
    ref = initialize_firebase()
    
    while True:
        # Capture face
        print("\nCapturing new face...")
        frame = capture_face()
        
        if frame is None:
            choice = input("Do you want to try again? (y/n): ")
            if choice.lower() != 'y':
                break
            continue
        
        # Get student details
        student_data = get_student_details()
        
        # Save image locally
        if not save_image(frame, student_data["id"]):
            print("Failed to save image")
            continue
        
        # Upload to Firebase Storage
        if not upload_to_storage(student_data["id"]):
            print("Failed to upload image to Firebase Storage")
            continue
        
        # Store in Firebase Database
        if not store_in_database(ref, student_data):
            print("Failed to store data in Firebase Database")
            continue
        
        # Update encodings
        if not update_encodings():
            print("Failed to update encodings")
            continue
        
        print(f"\nStudent {student_data['name']} (ID: {student_data['id']}) registered successfully!")
        
        # Ask if user wants to register another student
        choice = input("\nDo you want to register another student? (y/n): ")
        if choice.lower() != 'y':
            break
    
    print("Face Registration System closed")

if __name__ == "__main__":
    main()