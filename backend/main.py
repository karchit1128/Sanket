from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from avatar_api import router as avatar_router
from camera_api import router as camera_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Sanket ISL Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sanket API"}

app.include_router(avatar_router)
app.include_router(camera_router)
