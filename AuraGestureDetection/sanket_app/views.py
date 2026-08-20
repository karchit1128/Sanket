import os
import cv2
import time
import json
import threading
import datetime
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Import custom detection modules (from root directory)
from detector.gesture_detector import GestureDetector
from utils.gesture_stabilizer import GestureStabilizer
from utils.voice_engine import BackgroundVoiceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logger = logging.getLogger(__name__)

# Ensure assets directory exists for screenshots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'static', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(BASE_DIR, 'sanket_app', 'hand_landmarker.task')

# Hand connections for drawing (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

class LandmarkWrapper:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class HandLandmarksWrapper:
    def __init__(self, landmark_list):
        self.landmark = [LandmarkWrapper(lm.x, lm.y, getattr(lm, 'z', 0.0)) for lm in landmark_list]

class WebcamManager:
    def __init__(self):
        self.cap = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # State variables
        self.latest_frame = None
        self.latest_raw_frame = None
        self.fps = 0
        self.has_hand = False
        self.active_gesture = "Unknown"
        self.confidence = 0.0
        
        # Settings
        self.voice_enabled = False  # Disabled by default on server
        
        # Core engines
        self.detector = GestureDetector()
        self.stabilizer = GestureStabilizer(
            window_size=6,
            min_confidence=0.60,
            lock_consecutive_frames=3,
            cooldown_seconds=0.3
        )
        self.voice_engine = BackgroundVoiceEngine()
        
        # MediaPipe initialization using Tasks API (compatible with all MediaPipe 0.10.x & 1.0.x versions)
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
            logger.info("Initialized MediaPipe HandLandmarker Tasks API successfully.")
        except Exception as e:
            logger.warning(f"Tasks API init failed: {e}. Trying solutions fallback...")
            self.use_tasks_api = False
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.70,
                min_tracking_confidence=0.70
            )

    def start(self):
        """Starts the camera acquisition thread."""
        with self.lock:
            if self.running:
                return True
                
            logger.info("Starting Webcam Acquisition Thread...")
            
            # Direct open with CAP_DSHOW on index 0 for Windows
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                logger.warning("CAP_DSHOW failed on index 0, trying CAP_ANY fallback...")
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                logger.warning("Index 0 failed, trying index 1...")
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                
            if not self.cap.isOpened():
                logger.error("Failed to open any webcam device.")
                self.cap = None
                return False
                
            # Set camera dimensions for optimal speed/performance balance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True

    def stop(self):
        """Stops the thread and releases the webcam."""
        with self.lock:
            if not self.running:
                return
                
            logger.info("Stopping Webcam Acquisition Thread...")
            self.running = False
            
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.stabilizer.clear()
            self.active_gesture = "Unknown"
            self.confidence = 0.0
            self.fps = 0
            self.has_hand = False
            self.latest_frame = None
            self.latest_raw_frame = None
            
        logger.info("Webcam successfully closed and resources released.")

    def _capture_loop(self):
        """Background frame reading and processing loop."""
        prev_time = time.time()
        
        while self.running:
            with self.lock:
                cap_ref = self.cap
                
            if cap_ref is None:
                break
                
            ret, frame = cap_ref.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            # Keep raw frame in memory for clean screenshots (no landmarks)
            self.latest_raw_frame = frame.copy()
            
            # Mirror the frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert color space for MediaPipe (BGR -> RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            raw_gesture = "No Hand"
            raw_confidence = 0.0
            hand_present = False
            
            if self.use_tasks_api:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.landmarker.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    hand_present = True
                    raw_landmarks = detection_result.hand_landmarks[0]
                    hand_landmarks = HandLandmarksWrapper(raw_landmarks)
                    
                    # Draw custom neon hand skeleton
                    for connection in HAND_CONNECTIONS:
                        pt1 = (int(raw_landmarks[connection[0]].x * w), int(raw_landmarks[connection[0]].y * h))
                        pt2 = (int(raw_landmarks[connection[1]].x * w), int(raw_landmarks[connection[1]].y * h))
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
                        
                    for lm in raw_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (240, 240, 60), -1)
                        
                    # Extract handedness
                    handedness = "Right"
                    if detection_result.handedness:
                        try:
                            handedness = detection_result.handedness[0][0].category_name
                        except:
                            handedness = "Right"
                            
                    raw_gesture, raw_confidence = self.detector.detect_gesture(hand_landmarks, handedness)
            else:
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    hand_present = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    try:
                        handedness = results.multi_handedness[0].classification[0].label
                    except:
                        handedness = "Right"
                    raw_gesture, raw_confidence = self.detector.detect_gesture(hand_landmarks, handedness)
            
            # Apply stabilizer rules
            locked_gest, smoothed_conf, was_updated = self.stabilizer.add_prediction(raw_gesture, raw_confidence)
            
            # Update background variables
            with self.lock:
                self.has_hand = hand_present
                self.active_gesture = locked_gest
                self.confidence = smoothed_conf
                
                # Handle text-to-speech triggers (if server-side TTS enabled)
                if was_updated and self.voice_enabled and locked_gest not in ["No Hand", "Unknown"]:
                    self.voice_engine.speak(locked_gest)
                
                # Draw the gesture name directly on the video feed corner for reassurance
                if hand_present and locked_gest not in ["No Hand", "Unknown"]:
                    cv2.putText(
                        frame, 
                        f"Gesture: {locked_gest} ({int(smoothed_conf*100)}%)", 
                        (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (255, 60, 160),  # Neon pink/magenta
                        2, 
                        cv2.LINE_AA
                    )
                
                # Encode final processed frame to JPEG bytes
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.latest_frame = buffer.tobytes()
                
            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            prev_time = curr_time
            if time_diff > 0:
                self.fps = round(1.0 / time_diff, 1)
                
            # Prevent excessive CPU spinning
            time.sleep(0.005)

    def get_frame(self):
        """Returns the latest JPEG buffer."""
        with self.lock:
            return self.latest_frame

    def get_screenshot(self):
        """Saves current raw (unmarked) mirrored frame to assets folder."""
        with self.lock:
            if self.latest_raw_frame is None:
                return None
            frame = cv2.flip(self.latest_raw_frame, 1)  # keep mirrored for standard visual matches
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(ASSETS_DIR, filename)
        
        # Save frame
        cv2.imwrite(filepath, frame)
        return f"/static/assets/{filename}"

    def get_data(self):
        """Returns current state metrics."""
        with self.lock:
            return {
                "gesture": self.active_gesture,
                "confidence": round(self.confidence * 100, 1),
                "fps": self.fps if self.running else 0.0,
                "has_hand": self.has_hand,
                "history": self.stabilizer.get_history(),
                "is_camera_active": self.running,
                "voice_enabled": self.voice_enabled
            }

# Create WebcamManager Singleton instance lazily
_camera_manager = None
_camera_manager_lock = threading.Lock()

def get_camera_manager():
    global _camera_manager
    if _camera_manager is None:
        with _camera_manager_lock:
            if _camera_manager is None:
                _camera_manager = WebcamManager()
    return _camera_manager

def index(request):
    """Renders main premium HTML panel."""
    return render(request, 'index.html')

def gen_video(manager):
    """Generator function that yields live processed frames."""
    while True:
        frame_bytes = manager.get_frame()
        if frame_bytes is None:
            time.sleep(0.1)
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # cap yield at roughly 30 FPS

def video_feed(request):
    """Streams live MJPEG camera feed."""
    return StreamingHttpResponse(
        gen_video(get_camera_manager()),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def prediction_data(request):
    """Returns real-time prediction telemetry."""
    return JsonResponse(get_camera_manager().get_data())

@csrf_exempt
def toggle_camera(request):
    """Starts or stops the webcam feed."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except Exception:
            data = {}
        action = data.get('action')
        
        manager = get_camera_manager()
        if action == 'start':
            success = manager.start()
            return JsonResponse({"status": "started" if success else "failed"})
        elif action == 'stop':
            manager.stop()
            return JsonResponse({"status": "stopped"})
        else:
            if manager.running:
                manager.stop()
                return JsonResponse({"status": "stopped"})
            else:
                success = manager.start()
                return JsonResponse({"status": "started" if success else "failed"})
    return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

@csrf_exempt
def toggle_voice(request):
    """Toggles backend pyttsx3 speech manager."""
    if request.method == 'POST':
        manager = get_camera_manager()
        manager.voice_enabled = not manager.voice_enabled
        logger.info(f"Server-side TTS Speech toggled to: {manager.voice_enabled}")
        return JsonResponse({"voice_enabled": manager.voice_enabled})
    return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

@csrf_exempt
def capture_screenshot(request):
    """Captures high resolution frame and returns url."""
    if request.method == 'POST':
        manager = get_camera_manager()
        img_url = manager.get_screenshot()
        if img_url:
            return JsonResponse({"success": True, "img_url": img_url})
        return JsonResponse({"success": False, "error": "Camera not active or frame empty."})
    return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

@csrf_exempt
def speak(request):
    """Custom endpoint to speak raw message using backend engine."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except Exception:
            data = {}
        text = data.get('text', '')
        if text:
            manager = get_camera_manager()
            manager.voice_engine.speak(text)
            return JsonResponse({"success": True})
        return JsonResponse({"success": False, "error": "No text provided."})
    return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

# Cleanup hooks when shutting down server
def cleanup():
    global _camera_manager
    if _camera_manager is not None:
        logger.info("Cleaning up backend thread services...")
        _camera_manager.stop()
        _camera_manager.voice_engine.shutdown()

import atexit
atexit.register(cleanup)
