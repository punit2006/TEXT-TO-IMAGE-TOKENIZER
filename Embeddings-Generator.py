import torch
from .config import BERT_MODEL
from transformers import AutoModel
from .text_preprocessing import tokenize_text

# Load pre-trained model
model = AutoModel.from_pretrained(BERT_MODEL)

def preprocess_text(text_descriptions):
    """Generate embeddings for text descriptions"""
    inputs = tokenize_text(text_descriptions)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings
