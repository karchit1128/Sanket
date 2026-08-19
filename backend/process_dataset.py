#!/usr/bin/env python3
import os, cv2, numpy as np, argparse, shutil
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from feature_extraction import process_video

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
    print(f'Found {len(classes)} classes. Extracting features...')
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holistic_landmarker.task')
    if not os.path.exists(model_path):
        print(f'ERROR: holistic_landmarker.task not found at {model_path}')
        print('Run: curl -o backend/holistic_landmarker.task https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task')
        return
        
    landmarker = vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.IMAGE)
    )
    
    total = 0
    skipped = 0
    
    for class_name, class_path in classes:
        out_path = os.path.join(output_path, class_name)
        os.makedirs(out_path, exist_ok=True)
        videos = sorted([f for f in os.listdir(class_path) if f.endswith(VIDEO_EXTS)])
        print(f'  Processing [{class_name}]: {len(videos)} videos')
        
        for seq_idx, vf in enumerate(videos):
            cap = cv2.VideoCapture(os.path.join(class_path, vf))
            window = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                window.append(res)
                
            cap.release()
            
            features = process_video(window, target_frames=30)
            
            if features is not None:
                # Save as a single .npy file of shape (30, 546)
                np.save(os.path.join(out_path, f'{seq_idx}.npy'), features)
                total += 1
            else:
                print(f"    Skipping {vf} (No hands detected in sequence)")
                skipped += 1
                
        print(f'    Done: {len(videos)} sequences processed ({skipped} skipped)')
        
    zip_base = os.path.join(os.path.dirname(os.path.abspath(output_path)), 'dataset')
    shutil.make_archive(zip_base, 'zip', output_path)
    
    print(f'DONE! {total} total sequences saved successfully. {skipped} sequences skipped due to missing hands.')
    print(f'Upload to Colab: {zip_base}.zip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Root folder with class video folders')
    parser.add_argument('--output', default='dataset', help='Output folder for .npy files')
    args = parser.parse_args()
    process_dataset(args.dataset, args.output)
