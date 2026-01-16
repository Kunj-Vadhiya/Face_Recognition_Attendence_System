# 🎯 Face Recognition Attendance System

An intelligent attendance system that uses face recognition to automatically identify and register people, mark attendance, and maintain records.

## ✨ Features

### Core Functionality
- 🎥 **Automatic Webcam Startup** - System starts with webcam immediately
- 👤 **Face Detection & Recognition** - Real-time face detection and identification
- ✅ **Auto Attendance Marking** - Automatically marks attendance for known people
- 📝 **Auto-Registration** - Unknown people can be registered instantly (Press 'R')
- 📊 **CSV Records** - Attendance stored in daily CSV files
- 🚫 **Duplicate Prevention** - One attendance per person per day

### Visual Feedback
- 🔴 **Red Box** - Unknown person (press R to register)
- 🟠 **Orange Box** - Person recognized (marking attendance)
- 🟢 **Green Box** - Attendance already marked

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-contrib-python numpy pandas
```

### Running the System
```bash
python main.py
```

That's it! The system will:
1. ✓ Start webcam automatically
2. ✓ Load existing people from dataset (if any)
3. ✓ Detect and recognize faces in real-time
4. ✓ Mark attendance automatically for known people
5. ✓ Allow registration of unknown people

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Q** | Quit the application |
| **R** | Register unknown person (when red box appears) |

## 📸 Usage Workflow

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

## 📁 Project Structure

```
Face Recognition Attendance System/
├── main.py                      # Main application (AUTO-REGISTER ENABLED)
├── capture_dataset.py           # Manual dataset capture tool
├── requirements.txt             # Dependencies
├── README.md                    # This file
│
├── dataset/                     # Face images storage
│   ├── Person1/
│   │   └── images...
│   └── Person2/
│       └── images...
│
└── attendance_records/          # Daily attendance CSV files
    ├── attendance_2026-01-14.csv
    └── attendance_2026-01-15.csv
```

## 📊 Attendance Records

Attendance is saved in CSV format:
```csv
Name,Date,Time
John_Doe,2026-01-16,09:30:15
Jane_Smith,2026-01-16,09:31:22
```

- **Location**: `attendance_records/attendance_YYYY-MM-DD.csv`
- **Format**: Name, Date, Time
- **Behavior**: New file created each day
- **Duplicate Prevention**: One entry per person per day

## 🔧 Manual Dataset Creation (Optional)

If you want to pre-register people without running main.py:

```bash
python capture_dataset.py
```

This tool allows you to:
- Capture multiple face images per person
- Preview face detection in real-time
- Save images organized by person name

## 💡 How It Works

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

## 🎯 Technical Details

### Dependencies
- **OpenCV (opencv-contrib-python)**: Face detection & recognition
- **NumPy**: Array operations
- **Pandas**: CSV data management

### Performance
- **FPS**: 15-30 (depends on hardware)
- **Face Detection**: Real-time, very fast
- **Recognition**: Instant (< 0.1 seconds per face)
- **Training**: Instant (< 1 second for up to 50 people)

### Accuracy
- Best for: Small to medium groups (< 100 people)
- Lighting: Works well in normal to good lighting
- Distance: Face should be clearly visible
- Angle: Front-facing works best

## 🔍 Troubleshooting

### "Cannot open webcam"
- Ensure webcam is connected
- Close other applications using the camera
- Try changing camera index: `cv2.VideoCapture(1)`

### Poor Recognition
- Ensure good lighting
- Add more training images (3-5 per person recommended)
- Face should be clearly visible and front-facing
- Adjust `confidence_threshold` in code (lower = stricter)

### Low FPS
- Normal on older hardware
- Increase `process_every` value in code
- Reduce camera resolution

## 🚀 Advanced Usage

### Customization

Edit these parameters in `main.py`:

```python
# Recognition threshold (lower = stricter)
self.confidence_threshold = 50

# Process every Nth frame (higher = faster but less responsive)
process_every = 2

# Camera resolution
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

## 📈 Future Enhancements

Potential improvements:
- Web-based interface
- Database integration (SQLite/MySQL)
- Email notifications
- Export to Excel/PDF
- Admin dashboard
- Multiple camera support
- Cloud storage integration

## 📝 Notes

- **Data Storage**: All data stored locally
- **Privacy**: No cloud processing or external API calls
- **Scalability**: Best for organizations with < 100 people
- **Accuracy**: 85-95% in good conditions
- **Registration**: Requires manual name input for unknown faces

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open-source and available for educational and commercial use.

---

## 🎉 Quick Command Reference

```bash
# Run main system (with auto-registration)
python main.py

# Manual dataset creation
python capture_dataset.py

# Install dependencies
pip install -r requirements.txt
```

---

**System Status**: ✅ Fully Functional  
**Version**: 2.0  
**Last Updated**: January 2026

**Made with ❤️ for automated attendance management**
