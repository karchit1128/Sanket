from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from llm import translate_to_isl_gloss
import time

router = APIRouter()

# Simple In-Memory Rate Limiter (IP-based)
RATE_LIMIT = 10
TIME_WINDOW = 300 # 5 minutes
ip_tracker = {}

def check_rate_limit(ip: str):
    current_time = time.time()
    if ip not in ip_tracker:
        ip_tracker[ip] = []
    
    # Filter old timestamps
    ip_tracker[ip] = [t for t in ip_tracker[ip] if current_time - t < TIME_WINDOW]
    
    if len(ip_tracker[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 requests per 5 minutes.")
    
    ip_tracker[ip].append(current_time)

class TranslationRequest(BaseModel):
    text: str

@router.post("/translate")
def translate(req: TranslationRequest, request: Request):
    if request.client:
        check_rate_limit(request.client.host)

    if not req.text or req.text.strip() == "":
        return {"original": req.text, "gloss": ""}
        
    gloss = translate_to_isl_gloss(req.text)
    return {"original": req.text, "gloss": gloss}
