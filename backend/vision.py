import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import os
from feature_extraction import process_video

class ASLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = torch.max(x, dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class ASLDetector:
    def __init__(self):
        self.sequence = [] # Will store MediaPipe results objects
        self.frame_counter = 0
        self.last_result = None
        self.is_signing = False
        self.current_preds = []
        self.result_timer = 0

        base_options = python.BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), 'holistic_landmarker.task'))
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

        model_path = os.path.join(os.path.dirname(__file__), 'action.pt')
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            self.actions = np.array(checkpoint['classes'])
            num_classes = len(self.actions)
            self.model = ASLModel(input_size=546, hidden_size=64, num_classes=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f'Loaded action.pt: {num_classes} classes -> {self.actions.tolist()}')
        except Exception as e:
            print(f'WARNING: Could not load action.pt. Error: {e}')
            self.actions = np.array(['hello', 'thanks', 'iloveyou'])
            self.model = None

    def process_frame(self, image_np):
        self.frame_counter += 1
        # Horizontally flip the incoming selfie-camera frame to un-mirror it
        image_np = cv2.flip(image_np, 1)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.landmarker.detect(mp_image)

        raw_landmarks = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks:
                raw_landmarks.append({'x': float(lm.x), 'y': float(lm.y)})
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks:
                raw_landmarks.append({'x': float(lm.x), 'y': float(lm.y)})

        hands_visible = False
        if results.left_hand_landmarks or results.right_hand_landmarks:
            hands_visible = True

        if hands_visible:
            if not self.is_signing:
                self.is_signing = True
                self.signing_frame_count = 0
                self.sequence = [] # Start fresh buffer for this gesture
            
            self.sequence.append(results)
            # Cap maximum sequence length to prevent infinite buffer if hands never go down (5 seconds at ~30fps)
            if len(self.sequence) > 150:
                self.sequence = self.sequence[-150:]
                
            self.signing_frame_count += 1
            self.result_timer = 0
            status = "Signing..."
            
            # Hide old result while signing
            self.last_result = None
            
        else:
            status = "Idle"
            if self.is_signing:
                # User just put hands down. Gesture is finished!
                self.is_signing = False
                
                # Predict ONCE on the full accumulated gesture sequence
                if self.model is not None and len(self.sequence) >= 5:
                    features = process_video(self.sequence, target_frames=50)
                    if features is not None:
                        input_tensor = torch.tensor(np.array([features]), dtype=torch.float32)
                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                            prediction_idx = torch.argmax(probabilities).item()
                            confidence = probabilities[prediction_idx].item()
                            
                            self.last_result = f"{self.actions[prediction_idx]}"
                            print(f"Gesture Finished -> Predicted: {self.actions[prediction_idx]} (Conf: {confidence:.2f})")
                
                # Keep the result on screen for a while
                self.result_timer = 30
                self.signing_frame_count = 0
                self.sequence = [] # Clear buffer
                
            if self.result_timer > 0:
                self.result_timer -= 1
            else:
                self.last_result = None

        return {
            "status": status,
            "prediction": self.last_result,
            "landmarks": raw_landmarks,
            "signing_frame_count": getattr(self, 'signing_frame_count', 0)
        }
