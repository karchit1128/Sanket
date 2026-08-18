import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
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
        self.actions = np.array(['hello', 'thanks', 'iloveyou'])
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Initialize PyTorch Model
        input_size = 1662
        hidden_size = 64
        num_layers = 3
        num_classes = len(self.actions)
        
        self.model = ASLModel(input_size, hidden_size, num_layers, num_classes)
        model_path = os.path.join(os.path.dirname(__file__), 'action.pt')
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            print("Successfully loaded action.pt PyTorch model!")
        except Exception as e:
            print(f"WARNING: Could not load action.pt model. Ensure you have trained the data first! Error: {e}")
            self.model = None

    def process_frame(self, image_np):
        image, results = mediapipe_detection(image_np, self.holistic)
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]
        
        if len(self.sequence) == 30 and self.model is not None:
            # Convert sequence to PyTorch tensor
            input_tensor = torch.tensor(np.array([self.sequence]), dtype=torch.float32)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                prediction_idx = torch.argmax(probabilities).item()
                confidence = probabilities[prediction_idx].item()
            
            if confidence > 0.8:
                return self.actions[prediction_idx]
        return None
