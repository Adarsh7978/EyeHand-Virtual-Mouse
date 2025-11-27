# EyeHand-Virtual-Mouse
This project is designed to help users who cannot speak, hear, or use traditional input devices. The system detects hand gestures for mouse control and tracks eye movements to allow users to draw or write on screen. It uses real-time computer vision technology to provide an easy and touch-free way to interact with computers.
## ✨ Features
- Hand gesture detection
- Virtual mouse control using hand
- Eye tracking for air-drawing
- Real-time webcam input
- Smooth cursor movement
- Assistive technology for disabled users
- Python + OpenCV + MediaPipe based
## 🎥 Demo
![Demo](images/demo.gif)

## 🛠 Tech Stack
- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

## 🚀 Installation

1. Clone this repo:
   git clone https://github.com/username/repo-name

2. Install dependencies:
   pip install opencv-python mediapipe pyautogui numpy

3. Run the program:
   python main.py

## 🧠 How It Works

### Hand Detection
- MediaPipe detects hand landmarks.
- Index finger position is tracked.
- Cursor moves according to finger coordinates.

### Virtual Mouse
- Clicking is done using finger distance logic.

### Eye Tracking Drawing
- Eye landmarks are detected.
- Iris center coordinates allow drawing.
- When the eye blinks → drawing stops/starts.

Camera Input → MediaPipe → Hand Tracking → Cursor/Click  
                                  → Eye Tracking → Drawing

## 🔮 Future Improvements
- Add voice commands
- Add gesture-based text typing
- Add an interface for disabled users
- Improve accuracy in low-light

## 🎯 Use Cases
- Virtual mouse for hands-free operation
- Assistive tool for people with disabilities
- Drawing using eye movement
- HCI research projects
- AI/ML college projects

- ## 👨‍💻 Author
Adarsh Mishra  
B.Tech Data Science  
OIST Bhopal
MIT License


