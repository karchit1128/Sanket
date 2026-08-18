import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

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

def draw_landmarks_on_image(rgb_image, detection_result):
    for hand_landmarks in [detection_result.left_hand_landmarks, detection_result.right_hand_landmarks]:
        if hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * rgb_image.shape[1])
                y = int(landmark.y * rgb_image.shape[0])
                cv2.circle(rgb_image, (x, y), 5, (0, 0, 255), -1)
                
    if detection_result.pose_landmarks:
        for landmark in detection_result.pose_landmarks:
            x = int(landmark.x * rgb_image.shape[1])
            y = int(landmark.y * rgb_image.shape[0])
            cv2.circle(rgb_image, (x, y), 3, (0, 255, 0), -1)

# Data Collection Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), 'dataset')
actions = ['Alright', 'Good evening', 'Good night', 'How are you', 'Pleased', 'Thank you']
no_sequences = 30
sequence_length = 30 # 30 frames per video (3 seconds total)

print("Available signs to record:")
for i, act in enumerate(actions):
    print(f"{i+1}. {act}")

print("")
choice = input("Enter the number of the sign you want to record (or 'all' for all): ")
if choice.strip().lower() == 'all':
    selected_actions = actions
else:
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(actions): raise ValueError()
        selected_actions = [actions[idx]]
    except:
        print("Invalid choice. Exiting.")
        exit()

# Create Folders
for action in selected_actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

model_path = os.path.join(os.path.dirname(__file__), 'holistic_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HolisticLandmarkerOptions(base_options=base_options, output_face_blendshapes=False)
landmarker = vision.HolisticLandmarker.create_from_options(options)

print("Starting Data Collection in 3 seconds. Get ready!")
time.sleep(3)

cap = cv2.VideoCapture(0)
frame_counter = 0

for action in selected_actions:
    print(f"\n--- Get ready for '{action}' ---")
    cv2.waitKey(2000)
    for sequence in range(no_sequences):
        frame_num = 0
        while frame_num < sequence_length:
            ret, frame = cap.read()
            if not ret: continue
            
            frame_counter += 1
            
            # 10 FPS extraction (1 frame every 3 ticks)
            if frame_counter % 3 != 0:
                display_frame = cv2.flip(frame, 1)
                if frame_num == 0:
                    cv2.putText(display_frame, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(display_frame, f'Collecting frames for {action} Video Number {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                # Wait before starting the sequence
                if frame_num == 0:
                    cv2.waitKey(2000) 
                continue
                
            # Process UNFLIPPED frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = landmarker.detect(mp_image)
            
            # Export keypoints
            keypoints = normalize_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, f'{sequence}_{frame_num}.npy')
            np.save(npy_path, keypoints)
            
            # Draw on display frame
            draw_landmarks_on_image(frame, results)
            display_frame = cv2.flip(frame, 1)
            cv2.putText(display_frame, f'Collecting frames for {action} Video Number {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Data Collection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            
            frame_num += 1

cap.release()
cv2.destroyAllWindows()
print("Data collection completed successfully! You can now run train_model.py")
