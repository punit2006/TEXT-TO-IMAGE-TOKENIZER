from .config import BERT_MODEL
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

def tokenize_text(text_descriptions):
    """Tokenize a list of text descriptions"""
    inputs = tokenizer(
        text_descriptions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    return inputs
