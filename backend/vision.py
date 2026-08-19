import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import os
import re

def normalize_keypoints(results):
    if results.pose_landmarks:
        nx = results.pose_landmarks[0].x
        ny = results.pose_landmarks[0].y
        nz = results.pose_landmarks[0].z
        # Calculate horizontal proxy (shoulder width)
        l_sh = results.pose_landmarks[11]
        r_sh = results.pose_landmarks[12]
        shoulder_width = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
        
        # Calculate vertical proxy (nose to neck distance)
        neck_x = (l_sh.x + r_sh.x) / 2
        neck_y = (l_sh.y + r_sh.y) / 2
        neck_dist = np.sqrt((nx - neck_x)**2 + (ny - neck_y)**2)
        
        # Robust scale factor (immune to turning sideways)
        scale = max(shoulder_width, neck_dist * 2.5)
        if scale < 0.01:
            scale = 1.0
    else:
        nx, ny, nz = 0.0, 0.0, 0.0
        scale = 1.0

    if results.pose_landmarks:
        pose = np.array([[(r.x-nx)/scale, (r.y-ny)/scale, (r.z-nz)/scale, float(getattr(r,'visibility',0.0) or 0.0)] for r in results.pose_landmarks]).flatten()
    else:
        pose = np.zeros(33*4)

    if results.left_hand_landmarks:
        lh = np.array([[(r.x-nx)/scale, (r.y-ny)/scale, (r.z-nz)/scale] for r in results.left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(21*3)

    if results.right_hand_landmarks:
        rh = np.array([[(r.x-nx)/scale, (r.y-ny)/scale, (r.z-nz)/scale] for r in results.right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(21*3)

    kp = np.concatenate([pose, lh, rh])
    return kp if kp.shape[0] == 258 else np.zeros(258)


class ASLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=258, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ASLDetector:
    def __init__(self):
        self.sequence = []
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
            self.model = ASLModel(258, 64, 3, num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f'Loaded action.pt: {num_classes} classes -> {self.actions.tolist()}')
        except Exception as e:
            print(f'WARNING: Could not load action.pt. Error: {e}')
            self.actions = np.array(['hello', 'thanks', 'iloveyou'])
            self.model = None

    def process_frame(self, image_np):
        self.frame_counter += 1
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.landmarker.detect(mp_image)
        keypoints = normalize_keypoints(results)

        raw_landmarks = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks:
                raw_landmarks.append({'x': float(lm.x), 'y': float(lm.y)})
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks:
                raw_landmarks.append({'x': float(lm.x), 'y': float(lm.y)})

        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        # Pad sequence if it's less than 30 frames
        seq = self.sequence.copy()
        while len(seq) < 30:
            seq.append(seq[-1] if len(seq) > 0 else np.zeros(258))

        hands_visible = np.sum(np.abs(keypoints[132:])) > 0.01

        if hands_visible:
            if not self.is_signing:
                self.is_signing = True
                self.signing_frame_count = 0
                self.current_preds.clear()
            
            self.signing_frame_count += 1
            self.result_timer = 0
            status = "Signing..."
            
            # Predict continuously to find the most confident frame
            if self.model is not None:
                input_tensor = torch.tensor(np.array([seq]), dtype=torch.float32)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    prediction_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[prediction_idx].item()
                    
                    self.current_preds.append((confidence, self.actions[prediction_idx]))
                    print(f"Real-time -> {self.actions[prediction_idx]} ({confidence:.2f})")
            
            # Hide old result while signing
            self.last_result = None
            
        else:
            status = "Idle"
            if self.is_signing:
                # User just put hands down. Gesture is finished!
                self.is_signing = False
                
                if len(self.current_preds) > 0:
                    # Pick the highest confidence prediction during the gesture
                    best_conf, best_pred = max(self.current_preds, key=lambda x: x[0])
                    self.last_result = f"{best_pred}"
                    print(f"Gesture Finished -> Predicted: {best_pred} (Conf: {best_conf:.2f})")
                    self.current_preds.clear()
                
                # Keep the result on screen for 20 frames (2 seconds at 10 FPS)
                self.result_timer = 20
                self.signing_frame_count = 0
                
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
