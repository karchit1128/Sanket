#!/usr/bin/env python3
import os, cv2, numpy as np, argparse, shutil
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SAMPLE_EVERY_N_FRAMES = 3
MAX_FRAMES = 30

def normalize_keypoints(results):
    if results.pose_landmarks:
        nx, ny, nz = results.pose_landmarks[0].x, results.pose_landmarks[0].y, results.pose_landmarks[0].z
    else:
        nx, ny, nz = 0.0, 0.0, 0.0
    pose = np.array([[r.x-nx, r.y-ny, r.z-nz, float(getattr(r,'visibility',0.0) or 0.0)] for r in results.pose_landmarks]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh   = np.array([[r.x-nx, r.y-ny, r.z-nz] for r in results.left_hand_landmarks]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[r.x-nx, r.y-ny, r.z-nz] for r in results.right_hand_landmarks]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    kp = np.concatenate([pose, lh, rh])
    return kp if kp.shape[0] == 258 else np.zeros(258)

def find_classes(dataset_path):
    VIDEO_EXTS = ('.mp4','.avi','.mov','.mkv','.MOV','.MP4')
    classes = []
    for item in sorted(os.listdir(dataset_path)):
        p = os.path.join(dataset_path, item)
        if not os.path.isdir(p): continue
        vids = [f for f in os.listdir(p) if f.endswith(VIDEO_EXTS)]
        if vids:
            classes.append((item, p))
        else:
            for sub in sorted(os.listdir(p)):
                sp = os.path.join(p, sub)
                if os.path.isdir(sp):
                    svids = [f for f in os.listdir(sp) if f.endswith(VIDEO_EXTS)]
                    if svids: classes.append((sub, sp))
    return classes

def process_dataset(dataset_path, output_path):
    VIDEO_EXTS = ('.mp4','.avi','.mov','.mkv','.MOV','.MP4')
    classes = find_classes(dataset_path)
    if not classes:
        print('No classes found! Check your --dataset path.'); return
    print(f'Found {len(classes)} classes. Extracting at 10 FPS with nose normalization...')
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holistic_landmarker.task')
    if not os.path.exists(model_path):
        print(f'ERROR: holistic_landmarker.task not found at {model_path}')
        print('Run: curl -o backend/holistic_landmarker.task https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task')
        return
    landmarker = vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.IMAGE)
    )
    total = 0
    for class_name, class_path in classes:
        out_path = os.path.join(output_path, class_name)
        os.makedirs(out_path, exist_ok=True)
        videos = sorted([f for f in os.listdir(class_path) if f.endswith(VIDEO_EXTS)])
        print(f'  Processing [{class_name}]: {len(videos)} videos')
        for seq_idx, vf in enumerate(videos):
            cap = cv2.VideoCapture(os.path.join(class_path, vf))
            raw_n, window = 0, []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if raw_n % SAMPLE_EVERY_N_FRAMES == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    window.append(normalize_keypoints(res))
                    if len(window) >= MAX_FRAMES: break
                raw_n += 1
            cap.release()
            if not window: continue
            while len(window) < MAX_FRAMES: window.append(window[-1].copy())
            for fi, kp in enumerate(window):
                np.save(os.path.join(out_path, f'{seq_idx}_{fi}.npy'), kp)
            total += 1
        print(f'    Done: {len(videos)} sequences saved')
    zip_base = os.path.join(os.path.dirname(os.path.abspath(output_path)), 'dataset')
    shutil.make_archive(zip_base, 'zip', output_path)
    print(f'DONE! {total} total sequences. Upload to Colab: {zip_base}.zip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Root folder with class video folders')
    parser.add_argument('--output', default='dataset', help='Output folder for .npy files')
    args = parser.parse_args()
    process_dataset(args.dataset, args.output)
