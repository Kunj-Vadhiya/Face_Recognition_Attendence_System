"""
Face Recognition Attendance System
- Automatically starts webcam
- Identifies known people and marks attendance
- Auto-registers unknown people with captured images
"""

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time


class FaceRecognitionAttendance:
    """Face Recognition Attendance System with auto-registration."""
    
    def __init__(self, dataset_path="dataset", attendance_dir="attendance_records"):
        """Initialize the face recognition system."""
        self.dataset_path = dataset_path
        self.attendance_dir = attendance_dir
        
        # Create directories
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(attendance_dir, exist_ok=True)
        
        # Initialize face detector and recognizer
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage
        self.face_names = {}
        self.label_counter = 0
        self.trained = False
        
        # Attendance tracking
        self.today_date = datetime.now().strftime("%Y-%m-%d")
        self.attendance_file = os.path.join(
            attendance_dir, f"attendance_{self.today_date}.csv"
        )
        self.marked_today = set()
        
        # Recognition parameters
        self.confidence_threshold = 50
        self.unknown_faces = {}  # Track unknown faces for registration
        self.registration_frames = 10  # Frames needed for registration
        
        # Load existing data
        self.load_dataset()
        self._load_today_attendance()
    
    def _load_today_attendance(self):
        """Load existing attendance for today."""
        if os.path.exists(self.attendance_file):
            try:
                df = pd.read_csv(self.attendance_file)
                if 'Name' in df.columns:
                    self.marked_today = set(df['Name'].tolist())
                    print(f"Loaded {len(self.marked_today)} attendance records for today")
            except Exception as e:
                print(f"Error loading attendance: {e}")
    
    def load_dataset(self):
        """Load and train model with existing dataset."""
        print("\n" + "="*60)
        print("Loading Face Dataset")
        print("="*60)
        
        faces = []
        labels = []
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            print("Dataset folder created. No existing data.")
            return
        
        # Load existing person folders
        person_folders = [f for f in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, f)) 
                         and not f.startswith('.')]
        
        if len(person_folders) == 0:
            print("No existing people in dataset. Ready for auto-registration.")
            return
        
        for person_name in person_folders:
            person_dir = os.path.join(self.dataset_path, person_name)
            print(f"Loading: {person_name}")
            
            self.face_names[self.label_counter] = person_name
            
            for image_name in os.listdir(person_dir):
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                image_path = os.path.join(person_dir, image_name)
                
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(detected_faces) > 0:
                        x, y, w, h = detected_faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        
                        faces.append(face_roi)
                        labels.append(self.label_counter)
                    
                except Exception as e:
                    print(f"  Error processing {image_name}: {e}")
            
            self.label_counter += 1
        
        # Train if we have data
        if len(faces) > 0:
            print(f"\nTraining model with {len(faces)} images from {len(self.face_names)} people...")
            self.face_recognizer.train(faces, np.array(labels))
            self.trained = True
            print("✓ Model trained successfully!")
        else:
            print("No training data. Ready for auto-registration.")
        
        print("="*60 + "\n")
    
    def register_new_person(self, face_image, face_roi_gray):
        """Register a new person by capturing their face."""
        print("\n" + "="*60)
        print("NEW PERSON DETECTED - AUTO REGISTRATION")
        print("="*60)
        
        # Get name via console (in real app, could be via GUI or ID card)
        person_name = input("\nEnter person's name (or press Enter to skip): ").strip()
        
        if not person_name:
            print("Registration cancelled.")
            return None
        
        # Clean name
        person_name = person_name.replace(" ", "_")
        
        # Create person directory
        person_dir = os.path.join(self.dataset_path, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save multiple images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the current face image
        img_path = os.path.join(person_dir, f"{person_name}_{timestamp}_1.jpg")
        cv2.imwrite(img_path, face_image)
        print(f"✓ Saved image: {img_path}")
        
        # Add to training data
        label = self.label_counter
        self.face_names[label] = person_name
        self.label_counter += 1
        
        # Retrain model
        self.retrain_model_with_new_person(person_dir, label)
        
        print(f"✓ {person_name} registered successfully!")
        print("="*60 + "\n")
        
        return person_name
    
    def retrain_model_with_new_person(self, person_dir, label):
        """Retrain model including the new person."""
        # Load all existing training data
        faces = []
        labels = []
        
        # Load from all person folders
        for person_label, person_name in self.face_names.items():
            person_path = os.path.join(self.dataset_path, person_name)
            
            for img_name in os.listdir(person_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(detected_faces) > 0:
                        x, y, w, h = detected_faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        
                        faces.append(face_roi)
                        labels.append(person_label)
        
        # Retrain
        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))
            self.trained = True
    
    def mark_attendance(self, name):
        """Mark attendance for a person."""
        if name in self.marked_today:
            return False
        
        now = datetime.now()
        record = {
            'Name': name,
            'Date': now.strftime("%Y-%m-%d"),
            'Time': now.strftime("%H:%M:%S")
        }
        
        try:
            if os.path.exists(self.attendance_file):
                df = pd.read_csv(self.attendance_file)
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            else:
                df = pd.DataFrame([record])
            
            df.to_csv(self.attendance_file, index=False)
            self.marked_today.add(name)
            
            print(f"✓ Attendance marked: {name} at {record['Time']}")
            return True
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def start(self):
        """Start the attendance system with webcam."""
        print("\n" + "="*60)
        print("FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*60)
        print("Features:")
        print("  - Auto-identify known people")
        print("  - Auto-register unknown people")
        print("  - Automatic attendance marking")
        print("="*60)
        print(f"Known People: {len(self.face_names)}")
        print(f"Today's Attendance: {len(self.marked_today)}")
        print("="*60 + "\n")
        
        # Start webcam
        print("Starting webcam...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("ERROR: Cannot open webcam!")
            return
        
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Webcam started successfully!")
        print("\nControls:")
        print("  Q - Quit")
        print("  R - Register unknown person")
        print("="*60 + "\n")
        
        # Processing variables
        fps = 0
        frame_time = time.time()
        frame_count = 0
        process_every = 2
        pending_registration = None
        
        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("Error capturing frame!")
                    break
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % process_every == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)
                    )
                    
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi_resized = cv2.resize(face_roi, (200, 200))
                        
                        name = "Unknown"
                        color = (0, 0, 255)  # Red
                        confidence = 100
                        is_marked = False
                        
                        # Try to recognize if model is trained
                        if self.trained:
                            label, conf = self.face_recognizer.predict(face_roi_resized)
                            
                            if conf < self.confidence_threshold:
                                name = self.face_names.get(label, "Unknown")
                                confidence = conf
                                is_marked = name in self.marked_today
                                
                                if is_marked:
                                    color = (0, 255, 0)  # Green
                                else:
                                    color = (255, 165, 0)  # Orange
                                    # Mark attendance
                                    self.mark_attendance(name)
                        
                        # Store unknown face for potential registration
                        if name == "Unknown":
                            pending_registration = (frame.copy(), face_roi_resized)
                        
                        # Draw rectangle and text
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.rectangle(frame, (x, y+h-70), (x+w, y+h), color, cv2.FILLED)
                        
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (x+6, y+h-45), font, 0.6, (255, 255, 255), 1)
                        
                        if name != "Unknown":
                            conf_text = f"Match: {100-confidence:.1f}%"
                            cv2.putText(frame, conf_text, (x+6, y+h-25), font, 0.5, (255, 255, 255), 1)
                        
                        status = "Present" if is_marked else ("Detected" if name != "Unknown" else "Press R to Register")
                        cv2.putText(frame, status, (x+6, y+h-6), font, 0.4, (255, 255, 255), 1)
                
                # Calculate FPS
                current_time = time.time()
                if current_time - frame_time >= 1.0:
                    fps = frame_count / (current_time - frame_time)
                    frame_time = current_time
                    frame_count = 0
                
                # Display stats overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (380, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "Face Recognition Attendance", (20, 35), font, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Known: {len(self.face_names)} | Today: {len(self.marked_today)}", (20, 85), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Q: Quit | R: Register", (20, 105), font, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Face Recognition Attendance System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nShutting down...")
                    break
                elif key == ord('r') or key == ord('R'):
                    if pending_registration:
                        face_img, face_roi = pending_registration
                        new_name = self.register_new_person(face_img, face_roi)
                        if new_name:
                            # Mark attendance for newly registered person
                            self.mark_attendance(new_name)
                        pending_registration = None
                    else:
                        print("No unknown face detected. Please wait for detection.")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total People Known: {len(self.face_names)}")
            print(f"Attendance Marked Today: {len(self.marked_today)}")
            print(f"Attendance File: {self.attendance_file}")
            print("="*60 + "\n")


if __name__ == "__main__":
    system = FaceRecognitionAttendance()
    system.start()
