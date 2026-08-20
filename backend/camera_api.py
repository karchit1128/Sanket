from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import base64

router = APIRouter()

detector = None

def get_detector():
    global detector
    if detector is None:
        try:
            from vision import ASLDetector
            detector = ASLDetector()
        except Exception as e:
            print(f"Failed to load ASLDetector: {e}")
            detector = False
    return detector if detector is not False else None

@router.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    det = get_detector()
    try:
        while True:
            data = await websocket.receive_text()
            
            if det and data.startswith('data:image/jpeg;base64,'):
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    result = det.process_frame(img)
                    if result:
                        await websocket.send_json(result)
                        
    except WebSocketDisconnect:
        print("Client disconnected")

