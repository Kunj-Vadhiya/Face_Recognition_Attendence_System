# Face Recognition Attendance System

An intelligent attendance system that uses face recognition to automatically identify and register people, mark attendance, and maintain records.

## Features

### Core Functionality
- **Automatic Webcam Startup** - System starts with webcam immediately
- **Face Detection & Recognition** - Real-time face detection and identification
- **Auto Attendance Marking** - Automatically marks attendance for known people
- **Auto-Registration** - Unknown people can be registered instantly (Press 'R')
- **CSV Records** - Attendance stored in daily CSV files
- **Duplicate Prevention** - One attendance per person per day

### Visual Feedback
- **Red Box** - Unknown person (press R to register)
- **Orange Box** - Person recognized (marking attendance)
- **Green Box** - Attendance already marked

## Controls
**Q** -> Quit the application 
**R** -> Register unknown person (when red box appears) 

##  Usage Workflow

### For Known People:
1. System detects face → Shows orange box
2. Recognizes person → Displays name
3. Marks attendance automatically → Box turns green
4. Shows "Present" status

### For Unknown People:
1. System detects face → Shows red box with "Unknown"
2. Press **R** key to register
3. Enter person's name in console
4. Face is captured and saved
5. Model is retrained automatically
6. Attendance marked immediately
7. Person is now recognized in future

## How It Works

### Face Detection
- Uses Haar Cascade Classifier
- Detects faces in real-time from webcam
- Minimum face size: 100x100 pixels

### Face Recognition
- Algorithm: LBPH (Local Binary Patterns Histograms)
- Confidence threshold: 50 (lower = stricter matching)
- Face images resized to 200x200 for consistency

### Auto-Registration
1. Unknown face detected
2. User presses 'R' key
3. System prompts for name via console
4. Face image captured and saved
5. Dataset updated with new person
6. Model retrained with all data
7. Person immediately recognizable

### Attendance Logic
- Marks attendance on first recognition
- Prevents duplicate entries for same day
- Continues to show "Present" status after marking

### Poor Recognition
- Ensure good lighting
- Add more training images (3-5 per person recommended)
- Face should be clearly visible and front-facing
- Adjust `confidence_threshold` in code (lower = stricter)

### Low FPS
- Normal on older hardware
- Increase `process_every` value in code
- Reduce camera resolution

**System Status**: ✅ Fully Functional  
**Version**: 2.0  
**Last Updated**: January 2026

**Made with ❤️ for automated attendance management**
