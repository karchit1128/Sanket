from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def translate_to_isl_gloss(text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[ERROR] GROQ_API_KEY NOT SET"
        
    client = Groq(api_key=api_key)
    
    prompt = f"You are a strict Indian Sign Language (ISL) translator.\nYour goal is to convert the following English/Hindi sentence into ISL Gloss.\n\nRules for ISL Gloss:\n1. Use Subject-Object-Verb (SOV) structure.\n2. Remove helper verbs (is, am, are, was, were) and articles (a, an, the).\n3. Keep output strictly to upper-case English words separated by spaces.\n4. DO NOT output anything other than the translated gloss.\n\nInput: {text}\nOutput:"
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a strict ISL Gloss translator. Output only the requested gloss."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )
    
    return response.choices[0].message.content.strip()
