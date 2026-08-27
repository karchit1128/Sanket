from fastapi import APIRouter
from pydantic import BaseModel
from llm import translate_to_isl_gloss

router = APIRouter()

class TranslationRequest(BaseModel):
    text: str

@router.post("/translate")
def translate(req: TranslationRequest):
    if not req.text or req.text.strip() == "":
        return {"original": req.text, "gloss": ""}
        
    gloss = translate_to_isl_gloss(req.text)
    return {"original": req.text, "gloss": gloss}

@router.get("/debug-env")
def debug_env():
    import os
    groq = os.getenv("GROQ_API_KEY")
    return {
        "has_groq": bool(groq),
        "groq_prefix": groq[:4] if groq else None
    }
