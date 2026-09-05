import os
# import cv2
import time
import json
import threading
import datetime
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Import custom detection modules
from detector.gesture_detector import GestureDetector
from utils.gesture_stabilizer import GestureStabilizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'static', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(BASE_DIR, 'sanket_app', 'hand_landmarker.task')

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


class GestureEngine:
    """
    Stateless gesture detection engine.
    Accepts individual frames (numpy arrays) and returns detection results.
    Works with browser-streamed frames — no server webcam needed.
    """
    def __init__(self):
        self.detector = GestureDetector()
        self.stabilizer = GestureStabilizer(
            window_size=4,
            min_confidence=0.55,
            lock_consecutive_frames=2,
            cooldown_seconds=0.2
        )

        self.history = []
        self.lock = threading.Lock()
        self.landmarker = None
        self.use_tasks_api = False

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
            logger.info("MediaPipe HandLandmarker Tasks API initialized successfully.")
        except Exception as e:
            logger.warning(f"Tasks API failed: {e}. Trying Solutions API fallback...")
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.70,
                    min_tracking_confidence=0.70
                )
                self.use_tasks_api = False
                logger.info("MediaPipe Solutions API initialized as fallback.")
            except Exception as e2:
                logger.error(f"All MediaPipe init failed: {e2}")

    def process_landmarks(self, hands_data, mode="words"):
        """
        Process landmarks directly received from browser MediaPipe JS.
        hands_data: list of dicts [{'landmarks': [...], 'handedness': 'Right'}]
        """
        if not hands_data or len(hands_data) == 0:
            raw_gesture = "No Hand"
            raw_confidence = 0.0
            hand_present = False
            landmarks_out = []
        else:
            hand_present = True
            
            # Extract Hand 1
            hand_1_data = hands_data[0]
            raw_landmarks_1 = [LandmarkWrapper(lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)) for lm in hand_1_data.get('landmarks', [])]
            hand_1_landmarks = HandLandmarksWrapper(raw_landmarks_1)
            handedness_1 = hand_1_data.get('handedness', 'Right')
            landmarks_out = [{"x": lm.x, "y": lm.y} for lm in raw_landmarks_1] # return primary hand points for legacy frontend draw
            
            # Extract Hand 2 if present
            hand_2_landmarks = None
            handedness_2 = None
            if len(hands_data) > 1:
                hand_2_data = hands_data[1]
                raw_landmarks_2 = [LandmarkWrapper(lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)) for lm in hand_2_data.get('landmarks', [])]
                hand_2_landmarks = HandLandmarksWrapper(raw_landmarks_2)
                handedness_2 = hand_2_data.get('handedness', 'Left')

            raw_gesture, raw_confidence = self.detector.detect_gesture(hand_1_landmarks, handedness_1, hand_2_landmarks, handedness_2, mode)

        with self.lock:
            locked_gest, smoothed_conf, was_updated = self.stabilizer.add_prediction(raw_gesture, raw_confidence)

            if was_updated and locked_gest not in ["No Hand", "Unknown"]:
                self.history.insert(0, {
                    "gesture": locked_gest,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                })
                self.history = self.history[:20]

        return {
            "gesture": locked_gest,
            "confidence": round(smoothed_conf * 100, 1),
            "has_hand": hand_present,
            "landmarks": landmarks_out,
            "history": self.history[:10],
            "was_updated": was_updated
        }

    def process_frame(self, frame_bytes, mode="words"):
        """
        Process a JPEG frame (bytes) received from the browser as fallback.
        Returns dict with gesture, confidence, landmarks, has_hand.
        """
        try:
            import cv2
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_present = False
            landmarks_out = []
            raw_gesture = "No Hand"
            raw_confidence = 0.0

            if hasattr(self, 'hands'):
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    hand_present = True
                    hand_lm = results.multi_hand_landmarks[0]
                    landmarks_out = [{"x": lm.x, "y": lm.y} for lm in hand_lm.landmark]

                    handedness = "Right"
                    try:
                        handedness = results.multi_handedness[0].classification[0].label
                    except:
                        pass
                    raw_gesture, raw_confidence = self.detector.detect_gesture(hand_lm, handedness, None, None, mode)

            with self.lock:
                locked_gest, smoothed_conf, was_updated = self.stabilizer.add_prediction(raw_gesture, raw_confidence)

                if was_updated and locked_gest not in ["No Hand", "Unknown"]:
                    self.history.insert(0, {
                        "gesture": locked_gest,
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                    })
                    self.history = self.history[:20]

            return {
                "gesture": locked_gest,
                "confidence": round(smoothed_conf * 100, 1),
                "has_hand": hand_present,
                "landmarks": landmarks_out,
                "history": self.history[:10],
                "was_updated": was_updated
            }

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return self._empty_result()

    def _empty_result(self):
        return {
            "gesture": "No Hand",
            "confidence": 0.0,
            "has_hand": False,
            "landmarks": [],
            "history": [],
            "was_updated": False
        }

    def get_history(self):
        with self.lock:
            return self.history[:10]
            
    def clear_history(self):
        with self.lock:
            self.history.clear()
            self.stabilizer.clear()


# Singleton
_engine = None
_engine_lock = threading.Lock()

def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = GestureEngine()
    return _engine


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def process_frame(request):
    # Receives either:
    # 1. JSON payload with `landmarks` detected by browser MediaPipe JS
    # 2. Raw JPEG image bytes / base64
    if request.method != 'POST':
        return JsonResponse({"error": "POST only"}, status=405)

    content_type = request.headers.get('Content-Type', '')

    if 'application/json' in content_type:
        try:
            data = json.loads(request.body.decode('utf-8'))
            hands_data = data.get('hands', [])
            mode = data.get('mode', 'words')
            result = get_engine().process_landmarks(hands_data, mode=mode)
            return JsonResponse(result)
        except Exception as e:
            logger.error(f"JSON parsing error in process_frame: {e}")
            return JsonResponse({"error": str(e)}, status=400)

    body = request.body
    if not body:
        return JsonResponse({"error": "Empty body"}, status=400)

    if body.startswith(b'data:image'):
        import base64
        try:
            header, encoded = body.split(b',', 1)
            body = base64.b64decode(encoded)
        except Exception as e:
            return JsonResponse({"error": f"Base64 decode failed: {e}"}, status=400)

    mode = request.GET.get('mode', 'words')
    result = get_engine().process_frame(body, mode)
    return JsonResponse(result)



@csrf_exempt
def prediction_data(request):
    # Returns current history/state (used for polling fallback).
    engine = get_engine()
    return JsonResponse({
        "gesture": "No Hand",
        "confidence": 0,
        "has_hand": False,
        "fps": 0,
        "history": engine.get_history(),
        "is_camera_active": False,
        "voice_enabled": False
    })


@csrf_exempt
def toggle_camera(request):
    # Stub - camera is now browser-side. Returns ok.
    return JsonResponse({"status": "ok"})


@csrf_exempt
def toggle_voice(request):
    return JsonResponse({"voice_enabled": False})


@csrf_exempt
def capture_screenshot(request):
    return JsonResponse({"success": False, "error": "Screenshots handled client-side."})


@csrf_exempt
def speak(request):
    return JsonResponse({"success": True})


@csrf_exempt
def clear_history(request):
    # Clears the gesture history.
    get_engine().clear_history()
    return JsonResponse({"status": "ok"})
