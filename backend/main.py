from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import translate_to_isl_gloss
from vision import ASLDetector
import os
import cv2
import numpy as np
import base64

load_dotenv()

app = FastAPI(title="Sanket ISL Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ASL Detector
detector = ASLDetector()

class TranslationRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sanket API"}

@app.post("/translate")
def translate(req: TranslationRequest):
    gloss = translate_to_isl_gloss(req.text)
    return {"original": req.text, "gloss": gloss}

@app.websocket("/ws/video")
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
