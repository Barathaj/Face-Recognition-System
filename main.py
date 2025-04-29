import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime, timedelta

# ----------------------- CONFIGURATION SECTION -----------------------
# Firebase configuration
cred = credentials.Certificate("service.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://bigvision-22c68-default-rtdb.firebaseio.com/",
    'storageBucket': "bigvision-22c68.firebasestorage.app"
})

# Camera settings
CAMERA_ID = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# UI positioning constants
WEBCAM_BOX_X = 137
WEBCAM_BOX_Y = 244
WEBCAM_WIDTH = 655
WEBCAM_HEIGHT = 380
PANEL_X = 899
PANEL_Y = 23
PANEL_WIDTH = 438
PANEL_HEIGHT = 714

# Timing constants (in frames, assuming ~25fps)
ACTIVE_DISPLAY_TIME = 5 * 25      # 5 seconds for "ACTIVE" screen
DETAILS_DISPLAY_TIME = 5 * 25     # 5 seconds for student details screen
MARKED_DISPLAY_TIME = 2 * 25      # 2 seconds for "MARKED" screen (green checkmark)
ALREADY_MARKED_DISPLAY_TIME = 3 * 25  # 3 seconds for "ALREADY MARKED" screen
COOLDOWN_PERIOD = 60              # 1 minute cooldown before allowing re-marking
DB_COOLDOWN_PERIOD = 30           # 30 seconds cooldown in database

# Mode Types:
# 0: Scanning/Active mode
# 1: Student details mode (photo, ID, name)
# 2: Marked mode (green checkmark)
# 3: Already marked mode
MODE_SCANNING = 0
MODE_DETAILS = 1
MODE_MARKED = 2
MODE_ALREADY_MARKED = 3

# ----------------------- INITIALIZATION SECTION -----------------------
# Initialize Firebase bucket
bucket = storage.bucket()

# Initialize camera
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

# Load background image
imgBackground = cv2.imread('resources/background2.png')

# Load mode images
folderModePath = 'resources/new_modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
print("Loading Encode File ...")
try:
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)
    encodeListKnown, studentIds = encodeListKnownWithIds
    print(f"Encode File Loaded - {len(encodeListKnown)} faces")
except Exception as e:
    print(f"Error loading encode file: {e}")
    exit(1)

# ----------------------- HELPER FUNCTIONS -----------------------
def draw_centered_text(image, text, box_x, box_y, box_w, box_h, font_scale=0.6, thickness=2):
    """Draw text centered in a box region"""
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x = box_x + (box_w - text_width) // 2
    y = box_y + (box_h + text_height) // 2 - 5  # fine-tuned for vertical alignment
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness)  # black color

def fetch_student_data(student_id):
    """Fetch student information and image from Firebase"""
    student_info = None
    student_img = None
    
    try:
        # Get student data from database
        student_info = db.reference(f'Students/{student_id}').get()
        if not student_info:
            print(f"No data found for student ID {student_id}")
            return None, None
            
        # Get student image from storage
        try:
            blob = bucket.get_blob(f'images/{student_id}.jpeg')
            if blob:
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                student_img = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                print(f"Successfully loaded student image for ID {student_id}")
            else:
                print(f"No image found for student ID {student_id}")
        except Exception as e:
            print(f"Error loading student image: {e}")
    
    except Exception as e:
        print(f"Error fetching student data: {e}")
    
    return student_info, student_img

def update_attendance(student_id, student_info):
    """Update attendance record in Firebase"""
    try:
        if 'last_attendance_time' in student_info:
            last_time = datetime.strptime(student_info['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
            seconds_elapsed = (datetime.now() - last_time).total_seconds()
            
            if seconds_elapsed > DB_COOLDOWN_PERIOD:
                # Update attendance data
                ref = db.reference(f'Students/{student_id}')
                student_info['total_attendance'] += 1
                ref.child('total_attendance').set(student_info['total_attendance'])
                ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(f"Updated attendance for student ID {student_id}")
                return True
            else:
                print(f"Student ID {student_id} marked too recently in database")
                return False
        else:
            print(f"No attendance data for student ID {student_id}")
            return False
    except Exception as e:
        print(f"Error updating attendance: {e}")
        return False

def display_student_info(img, student_info, student_img, student_id):
    """Display student information on the right panel"""
    try:
        # Display student ID
        if student_id is not None:
            draw_centered_text(img, str(student_id), 1025, 555, 100, -170)

        # Display student name
        if student_info and 'name' in student_info:
            draw_centered_text(img, student_info['name'], 1025, 555, 200, 40)

        # Display student image
        if student_img is not None and isinstance(student_img, np.ndarray) and student_img.size > 0:
            try:
                img_resized = cv2.resize(student_img, (216, 216))
                img[150:150 + 216, 1009:1009 + 216] = img_resized
            except Exception as e:
                print(f"Error displaying student image: {e}")
    except Exception as e:
        print(f"Error displaying student info: {e}")

def cleanup_marked_students(marked_dict, cooldown):
    """Remove expired entries from the marked students dictionary"""
    current_time = datetime.now()
    expired_ids = [student_id for student_id, mark_time in marked_dict.items() 
                  if (current_time - mark_time).total_seconds() > cooldown]
    
    for student_id in expired_ids:
        del marked_dict[student_id]

# ----------------------- MAIN PROGRAM -----------------------
# Initialize tracking variables
modeType = MODE_SCANNING
counter = 0
current_id = -1
imgStudent = None
studentInfo = None
marked_students = {}  # To track recently marked students {id: timestamp}
face_detected = False

print("Starting Face Attendance System...")

while True:
    # Capture frame from camera
    success, img = cap.read()
    if not success:
        print("Failed to get frame from camera")
        continue

    # Create a copy of the background for each frame
    imgBackground = cv2.imread('resources/background2.png')
    
    # Resize webcam feed to match the green box
    imgResized = cv2.resize(img, (WEBCAM_WIDTH, WEBCAM_HEIGHT))

    # Paste webcam feed onto background
    imgBackground[WEBCAM_BOX_Y:WEBCAM_BOX_Y+WEBCAM_HEIGHT, 
                 WEBCAM_BOX_X:WEBCAM_BOX_X+WEBCAM_WIDTH] = imgResized
    
    # Create a copy to work with for drawing
    imgBackgroundCopy = imgBackground.copy()

    # Update the right panel image based on current mode
    panelResized = cv2.resize(imgModeList[modeType], (PANEL_WIDTH, PANEL_HEIGHT))
    imgBackgroundCopy[PANEL_Y:PANEL_Y+PANEL_HEIGHT, 
                     PANEL_X:PANEL_X+PANEL_WIDTH] = panelResized

    # Process face detection only if we're in scanning mode
    if modeType == MODE_SCANNING:
        # Resize and convert image for face recognition (downscale for efficiency)
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        # Detect faces and encode them
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        
        # Reset face_detected flag
        face_detected = False
        
        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                if len(faceDis) > 0:  # Make sure we have valid face distances
                    matchIndex = np.argmin(faceDis)
                    
                    # If a known face is detected with good confidence
                    if matches[matchIndex] and faceDis[matchIndex] < 0.6:
                        # Draw rectangle around face
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        bbox = WEBCAM_BOX_X + x1, WEBCAM_BOX_Y + y1, x2 - x1, y2 - y1
                        imgBackgroundCopy = cvzone.cornerRect(imgBackgroundCopy, bbox, rt=0)
                        
                        # Get student ID
                        current_id = studentIds[matchIndex]
                        face_detected = True
                        
                        # Check if this student was recently marked (within cooldown period)
                        current_time = datetime.now()
                        if current_id in marked_students:
                            time_diff = (current_time - marked_students[current_id]).total_seconds()
                            
                            # If within cooldown period, show "Already Marked" screen
                            if time_diff < COOLDOWN_PERIOD:
                                counter = 1
                                modeType = MODE_ALREADY_MARKED
                                
                                # Update right panel immediately for "Already Marked"
                                panelResized = cv2.resize(imgModeList[modeType], (PANEL_WIDTH, PANEL_HEIGHT))
                                imgBackgroundCopy[PANEL_Y:PANEL_Y+PANEL_HEIGHT, 
                                                PANEL_X:PANEL_X+PANEL_WIDTH] = panelResized
                                
                                print(f"Student ID {current_id} already marked - showing 'Already Marked' screen")
                                continue
                        
                        # Fetch student data
                        studentInfo, imgStudent = fetch_student_data(current_id)
                        
                        if studentInfo:
                            # Update attendance in database
                            attendance_updated = update_attendance(current_id, studentInfo)
                            
                            if attendance_updated:
                                # Record this student as recently marked
                                marked_students[current_id] = datetime.now()
                                
                                # Switch to "Student Details" mode (photo, ID, name)
                                counter = 1
                                modeType = MODE_DETAILS
                                
                                # Update right panel immediately for "Student Details"
                                panelResized = cv2.resize(imgModeList[modeType], (PANEL_WIDTH, PANEL_HEIGHT))
                                imgBackgroundCopy[PANEL_Y:PANEL_Y+PANEL_HEIGHT, 
                                                PANEL_X:PANEL_X+PANEL_WIDTH] = panelResized
                            else:
                                # Student was marked too recently (database perspective)
                                counter = 1
                                modeType = MODE_ALREADY_MARKED
                                
                                # Update right panel immediately
                                panelResized = cv2.resize(imgModeList[modeType], (PANEL_WIDTH, PANEL_HEIGHT))
                                imgBackgroundCopy[PANEL_Y:PANEL_Y+PANEL_HEIGHT, 
                                                PANEL_X:PANEL_X+PANEL_WIDTH] = panelResized
    
    # Handle mode transitions based on counters
    if counter > 0:
        counter += 1
        
        # After showing student details screen for specified time, transition to marked screen
        if modeType == MODE_DETAILS and counter >= DETAILS_DISPLAY_TIME:
            print("Transitioning from details to marked mode")
            modeType = MODE_MARKED  # Switch to marked mode (green checkmark)
            counter = 1  # Reset counter for marked mode
            
            # Update the right panel immediately for marked
            panelResized = cv2.resize(imgModeList[modeType], (PANEL_WIDTH, PANEL_HEIGHT))
            imgBackgroundCopy[PANEL_Y:PANEL_Y+PANEL_HEIGHT, 
                             PANEL_X:PANEL_X+PANEL_WIDTH] = panelResized
        
        # When in student details mode, display student information
        if modeType == MODE_DETAILS:
            display_student_info(imgBackgroundCopy, studentInfo, imgStudent, current_id)
        
        # After showing "Already Marked" for specified time, return to scanning mode
        elif modeType == MODE_ALREADY_MARKED and counter >= ALREADY_MARKED_DISPLAY_TIME:
            counter = 0
            modeType = MODE_SCANNING
            studentInfo = None
            imgStudent = None
        
        # After showing "Marked" screen (checkmark) for specified time, return to scanning mode
        elif modeType == MODE_MARKED and counter >= MARKED_DISPLAY_TIME:
            counter = 0
            modeType = MODE_SCANNING
            studentInfo = None
            imgStudent = None
    
    # Clean up expired entries in marked_students dictionary
    cleanup_marked_students(marked_students, COOLDOWN_PERIOD)
    
    # Display the final image
    cv2.imshow("Face Attendance", imgBackgroundCopy)
    
    # Check for key press (ESC to exit)
    if cv2.waitKey(1) == 27:
        break

# Clean up resources before exiting
print("Shutting down Face Attendance System...")
cap.release()
cv2.destroyAllWindows()