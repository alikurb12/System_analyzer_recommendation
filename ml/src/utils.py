import re

def clean_text(text: str) -> str:
    text = text.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
    lines = [line.strip() for line in text.split('\n')]
    cleaned_text = '\n'.join(lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()