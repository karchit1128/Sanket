import sys
sys.path.append('.')
from llm import translate_to_isl_gloss

try:
    result = translate_to_isl_gloss('Hello, how are you?')
    print('SUCCESS:', result)
except Exception as e:
    print('ERROR:', e)
