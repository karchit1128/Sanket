from fastapi import APIRouter
from pydantic import BaseModel
from llm import translate_to_isl_gloss

router = APIRouter()

class TranslationRequest(BaseModel):
    text: str

@router.post("/translate")
def translate(req: TranslationRequest):
    gloss = translate_to_isl_gloss(req.text)
    return {"original": req.text, "gloss": gloss}
