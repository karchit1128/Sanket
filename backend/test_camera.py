import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from vision import ASLDetector

def draw_landmarks_on_image(rgb_image, detection_result):
    # Draw simple red circles for hands
    for hand_landmarks in [detection_result.left_hand_landmarks, detection_result.right_hand_landmarks]:
        if hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * rgb_image.shape[1])
                y = int(landmark.y * rgb_image.shape[0])
                cv2.circle(rgb_image, (x, y), 5, (0, 0, 255), -1)
                
    # Draw simple green circles for body/pose
    if detection_result.pose_landmarks:
        for landmark in detection_result.pose_landmarks:
            x = int(landmark.x * rgb_image.shape[1])
            y = int(landmark.y * rgb_image.shape[0])
            cv2.circle(rgb_image, (x, y), 3, (0, 255, 0), -1)

def main():
    print("Starting Camera... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    
    # Initialize our trained model detector
    detector = ASLDetector()
    
    # Initialize the Tasks API landmarker for visualization
    model_path = os.path.join(os.path.dirname(__file__), 'holistic_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HolisticLandmarkerOptions(base_options=base_options, output_face_blendshapes=False)
    landmarker = vision.HolisticLandmarker.create_from_options(options)

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
            
        frame_counter += 1
        
        # Limit OpenCV 30 FPS to 10 FPS to match training data (3 seconds = 30 frames)
        if frame_counter % 3 != 0:
            display_frame = cv2.flip(frame, 1)
            cv2.imshow("Camera & Gesture Test (Press Q to quit)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # 1. Run our actual ASL Detector for prediction on the UNFLIPPED frame
        result = detector.process_frame(frame)
        prediction = result.get('prediction')

        # 2. Run landmarker for drawing (so we can visually confirm hands are tracked)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = landmarker.detect(mp_image)
        
        draw_landmarks_on_image(frame, results)

        # Flip the final frame for display only (so it feels like a mirror to the user)
        display_frame = cv2.flip(frame, 1)
        
        if result.get('status') == "Signing...":
            frame_count = result.get('signing_frame_count', 0)
            progress = min(frame_count / 30.0, 1.0)
            
            cv2.putText(display_frame, "Analyzing Gesture...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            # Draw Progress Bar
            bar_width = 300
            bar_height = 15
            x, y = 10, 70
            cv2.rectangle(display_frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), -1)
            cv2.rectangle(display_frame, (x, y), (x + int(bar_width * progress), y + bar_height), (0, 165, 255), -1)
            
        elif prediction:
            cv2.putText(display_frame, f"Prediction: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        cv2.imshow("Camera & Gesture Test (Press Q to quit)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
