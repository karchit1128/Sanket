import sys
sys.path.append('.')
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': 'You are an ISL translator. Translate: hello'}],
        model='qwen/qwen3.6-27b',
    )
    print('SUCCESS:', response.choices[0].message.content)
except Exception as e:
    print('ERROR:', e)
