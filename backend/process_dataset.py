import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, getattr(res, 'visibility', 0.0) if getattr(res, 'visibility', None) is not None else 0.0] for res in results.pose_landmarks]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_dataset(dataset_dir, output_dir, sequence_length=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    words = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    base_options = python.BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), 'holistic_landmarker.task'))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for word in words:
            word_dir = os.path.join(dataset_dir, word)
            videos = [v for v in os.listdir(word_dir) if v.endswith('.mp4')]
            
            for seq_num, video in enumerate(videos):
                video_path = os.path.join(word_dir, video)
                cap = cv2.VideoCapture(video_path)
                
                out_seq_dir = os.path.join(output_dir, word, str(seq_num))
                os.makedirs(out_seq_dir, exist_ok=True)
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count < sequence_length:
                    print(f"Skipping {video_path} (too short: {frame_count} frames)")
                    continue
                
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    results = landmarker.detect(mp_image)
                    
                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(out_seq_dir, f"{frame_num}.npy"), keypoints)
                    
                cap.release()
                print(f"Processed: {word} - Video {seq_num}")

if __name__ == '__main__':
    print("Welcome to the ISL Dataset Processor!")
    dataset_path = input("Enter the absolute path to your downloaded dataset folder (e.g., C:/Downloads/INCLUDE): ")
    output_path = os.path.join(os.path.dirname(__file__), 'MP_Data')
    
    if os.path.exists(dataset_path):
        print("Processing dataset... This might take a while depending on the size.")
        process_dataset(dataset_path, output_path)
        print(f"Done! All data extracted to {output_path}")
    else:
        print("Error: Dataset path not found.")
