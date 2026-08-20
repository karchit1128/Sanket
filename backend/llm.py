from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def translate_to_isl_gloss(text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[ERROR] GROQ_API_KEY NOT SET"
        
    client = Groq(api_key=api_key)
    
    prompt = f"You are a strict Indian Sign Language (ISL) translator.\nYour goal is to convert the following English/Hindi sentence into ISL Gloss.\n\nRules for ISL Gloss:\n1. Use Subject-Object-Verb (SOV) structure.\n2. Remove helper verbs (is, am, are, was, were) and articles (a, an, the).\n3. Keep output strictly to upper-case English words separated by spaces.\n4. DO NOT output anything other than the translated gloss. NO explanations, NO intro, NO chain of thought.\n\nInput: {text}\nOutput:"
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict ISL Gloss translator. Output ONLY the translated words. Absolutely no conversational text."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"LLM Error: {e}")
        # Simple fallback conversion (uppercase words without basic stopwords)
        stopwords = {"is", "am", "are", "was", "were", "a", "an", "the", "to"}
        words = [w.upper() for w in text.split() if w.lower() not in stopwords]
        return " ".join(words)

