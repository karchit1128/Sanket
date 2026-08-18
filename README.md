# 🤟 Project Sanket (संकेत)
**Bidirectional Indian Sign Language (ISL) Translation Platform**

[![Next.js](https://img.shields.io/badge/Next.js-black?style=for-the-badge&logo=next.js&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00B2FF?style=for-the-badge&logo=google)](#)
[![React Three Fiber](https://img.shields.io/badge/React_Three_Fiber-black?style=for-the-badge&logo=react)](#)

Project Sanket is a production-ready, bidirectional communication tool designed to bridge the gap between the deaf and hearing communities in India. It captures spoken English/Hindi and renders an ISL 3D Avatar, while simultaneously capturing ISL via a standard webcam and converting it into spoken English/Hindi.

This project was architected for the **Smart India Hackathon (SIH260417)**.

---

## ✨ Key Features
*   **Bidirectional Translation:** Hearing $\rightarrow$ Deaf (Avatar) | Deaf $\rightarrow$ Hearing (Text-to-Speech).
*   **Bilingual Support (Hindi & English):** Integrates the Government of India's **Bhashini API** for robust Hindi ASR, alongside the Web Speech API for offline English translation.
*   **AI Grammar Smoothing:** Uses Groq/Llama-3 to convert broken ISL Gloss (e.g., `[I] [HEADACHE]`) into fluid conversational sentences.
*   **Progressive Web App (PWA):** Runs seamlessly on any laptop, tablet, or mobile phone browser without requiring heavy app installations.
*   **Fail-Safe Accessibility:** If the 3D avatar fails to load due to bandwidth, the UI automatically falls back to large, high-contrast ISL Gloss text.

---

## 🧠 System Architecture

Sanket utilizes a decoupled frontend/backend architecture to ensure real-time performance.

### 1. The "Ears" (Speech to ISL Gloss)
*   **Audio Capture:** Native browser `MediaRecorder`.
*   **ASR:** Bhashini API (Hindi) / Web Speech API (English).
*   **Translation:** LLM translates standard grammar to ISL Subject-Object-Verb (SOV) Gloss.

### 2. The "Mouth" (Gloss to 3D Avatar)
*   **Rendering:** React Three Fiber (`@react-three/drei`).
*   **Avatar:** High-fidelity `.glb` models generated via Ready Player Me.
*   **Animation:** Sequential triggering of `.fbx` animation clips. Fallback to fingerspelling (A-Z) for unknown words using LLM synonym mapping.

### 3. The "Eyes" (Camera to Text)
*   **Computer Vision:** Google MediaPipe Holistic extracts 21 `(X,Y,Z)` hand landmarks per frame.
*   **Classification:** A Sequential LSTM Neural Network (TensorFlow/Keras) analyzes 30-frame windows to predict the exact sign.
*   **UX:** Implements a "Push-to-Sign" mechanic to ensure 99% accuracy in noisy hackathon environments.



---

## 🚀 Local Setup & Installation

### Prerequisites
*   Node.js (v18+)
*   Python (3.9 - *Strict requirement for MediaPipe/TF compatibility*)
*   API Keys: Bhashini & Groq (or OpenAI)

### 1. Frontend Setup (Next.js)
```bash
# Clone the repo and navigate to frontend
cd frontend
npm install

# Run the development server (disable PWA in dev mode)
npm run dev
```

### 2. Backend Setup (FastAPI)
```bash
# Navigate to backend
cd backend

# Create a virtual environment (Python 3.9)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (Pin specific versions to avoid LSTM shape traps)
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload
```

---

## ⚠️ Crucial Hackathon Development Traps
If you are contributing to this project during a hackathon, read these warnings:
1.  **Shape Mismatches:** If the LSTM crashes with a `ValueError`, check `data.shape`. The ASL tutorial code expects 1662 features per frame. If your ISL dataset only uses hands (126 features), update the `input_shape` in Keras.
2.  **Avatar T-Pose Glitch:** If the avatar refuses to sign, the `.glb` bone names (e.g., `LeftArm`) do not match the `.fbx` animation bone names (e.g., `mixamorig:LeftArm`). Run the avatar through Adobe Mixamo to standardize the rig.
3.  **CORS Errors:** Ensure the FastAPI `CORSMiddleware` is set to `allow_origins=["*"]` during local development to allow Next.js to communicate with the Python backend.
