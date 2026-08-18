import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import os

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, getattr(res, 'visibility', 0.0) if getattr(res, 'visibility', None) is not None else 0.0] for res in results.pose_landmarks]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks]).flatten() if results.face_landmarks else np.zeros(478*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

class ASLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ASLModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ASLDetector:
    def __init__(self):
        self.sequence = []
        
        # Dynamically load actions
        actions_path = os.path.join(os.path.dirname(__file__), 'actions.txt')
        if os.path.exists(actions_path):
            with open(actions_path, 'r') as f:
                self.actions = np.array([line.strip() for line in f.readlines()])
        else:
            self.actions = np.array(['hello', 'thanks', 'iloveyou'])
            
        # Initialize MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), 'holistic_landmarker.task'))
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)
        
        # Initialize PyTorch Model
        input_size = 1692
        hidden_size = 64
        num_layers = 3
        num_classes = len(self.actions)
        
        self.model = ASLModel(input_size, hidden_size, num_layers, num_classes)
        model_path = os.path.join(os.path.dirname(__file__), 'action.pt')
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            self.model.eval()
            print(f"Successfully loaded action.pt PyTorch model for {num_classes} actions!")
        except Exception as e:
            print(f"WARNING: Could not load action.pt model. Error: {e}")
            self.model = None

    def process_frame(self, image_np):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results = self.landmarker.detect(mp_image)
        keypoints = extract_keypoints(results)
        
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]
        
        if len(self.sequence) == 30 and self.model is not None:
            input_tensor = torch.tensor(np.array([self.sequence]), dtype=torch.float32)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                prediction_idx = torch.argmax(probabilities).item()
                confidence = probabilities[prediction_idx].item()
            
            if confidence > 0.8:
                return self.actions[prediction_idx]
        return None
