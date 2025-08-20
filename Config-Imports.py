# -*- coding: utf-8 -*-
"""Config and common imports"""

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from diffusers import StableDiffusionPipeline
from PIL import Image

# Default model names
BERT_MODEL = "bert-base-uncased"
SD_MODEL = "CompVis/stable-diffusion-v1-4"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
