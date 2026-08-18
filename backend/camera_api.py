from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from vision import ASLDetector
import cv2
import numpy as np
import base64

router = APIRouter()

# Initialize ASL Detector once at startup
detector = ASLDetector()

@router.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            if data.startswith('data:image/jpeg;base64,'):
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    prediction = detector.process_frame(img)
                    if prediction:
                        await websocket.send_json({"prediction": prediction})
                        
    except WebSocketDisconnect:
        print("Client disconnected")
