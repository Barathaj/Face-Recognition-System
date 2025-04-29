# 📸 Face Attendance System

A modern, efficient facial recognition-based attendance tracking system that automates the process of recording attendance using computer vision technology.

---

## 📋 Overview

The **Face Attendance System** is designed to simplify and digitize the attendance marking process. The system uses advanced facial recognition techniques to identify individuals, mark their attendance, and maintain comprehensive attendance records in a Firebase database.

---


## ✨ Key Features

- **Automated Facial Recognition:** Uses state-of-the-art facial detection and recognition algorithms.
- **Two-Phase Implementation:**
  - **Phase 1:** Utilizes `face_recognition` library for encoding and detection.
  - **Phase 2:** Implements open-source deep learning models (MTCNN and FaceNet).
- **Real-time Database:** Firebase integration for secure and reliable data storage.
- **Duplicate Prevention:** Intelligent system to prevent duplicate attendance entries.
- **High Accuracy:** Robust face detection and matching capabilities.

---

## Demo :



https://github.com/user-attachments/assets/0c582f72-4d98-4fea-bd0d-8eb1b5a7b331




## 🛠️ Technical Architecture
![architecture](https://github.com/user-attachments/assets/1d827063-a9be-4ca5-b876-fb6c88b3ec86)

### 🔐 Registration Process

1. Capture user photo from the interface.
2. Extract and encode facial features.
3. Collect user details (name, ID, etc.).
4. Store encoded face data and personal information in Firebase.

### 🕒 Attendance Marking

1. Capture image from camera.
2. Detect and extract facial features.
3. Compare against stored face encodings.
4. Find the closest match through similarity search.
5. Retrieve user details from Firebase.
6. Mark attendance with timestamp.
7. Prevent re-marking attendance within 10-minute intervals.

---

## 💻 Implementation Details

### ✅ Phase 1

- Utilizes the `face_recognition` library for:
  - Facial encoding
  - Face detection

### 🚀 Phase 2

- Implements more advanced deep learning models:
  - **MTCNN (Multi-task Cascaded Convolutional Networks)** for face detection.
  - **FaceNet** for facial feature extraction and encoding.
- Custom similarity search algorithms for matching.

---

## 🔍 Accuracy

The system demonstrates **high accuracy** in recognizing registered individuals across various lighting conditions and angles. The Phase 2 implementation with **MTCNN** and **FaceNet** offers improved accuracy compared to the basic library-based approach.

---

## 🗄️ Database Structure (Firebase Realtime DB)

- **User Profiles**: Name, ID, and additional details.
- **Facial Encodings**: Stored for registered users.
- **Attendance Records**: With date and timestamp.
- **System Configuration Settings**: Stored for runtime and behavioral configurations.

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.7+
- Required libraries (see `requirements.txt`)
- Firebase account and project setup
- Webcam or camera device

### 🧰 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/face-attendance-system.git
cd face-attendance-system
```

Install dependencies:

```
pip install -r requirements.txt
```

Configure Firebase credentials (see Configuration).

Run the application:

```
python main.py
```

⚙️ Configuration
Set up Firebase project and generate admin SDK credentials.

Place the Firebase credentials JSON file in the root directory.

Update Firebase references and paths in the code (firebase_config.py or similar file).
