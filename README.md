# Text-to-Image Tokenizer using Transformers & Stable Diffusion

This repository demonstrates an easy pipeline to convert text prompts into embeddings with a transformer (BERT), generate images from text using Stable Diffusion, and compare prompt similarity with cosine metrics. It's perfect for experimenting with text-to-image workflows and prompt engineering.

**Google Colab Notebook:**  

👉 https://colab.research.google.com/drive/1_yoWpTLAobjjC3Wyz5Dgq8X3o4pLjHg9?usp=sharing

***

## Features

- **Text Embedding**: Uses BERT to encode text prompts into vector representations.
- **Cosine Similarity**: Compares the semantic similarity of different input prompts.
- **Image Generation**: Leverages Stable Diffusion for high-quality text-to-image generations.
- **Sample Outputs**: See generated images below.

***

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/text-to-image-tokenizer.git
cd text-to-image-tokenizer
pip install torch transformers diffusers scipy
```

Or just run in Colab for GPU support and no local install.

***

## Usage

Edit and run the provided Python script or notebook:
```bash
python 3_text_to_image_tokenizer.py
```
This will:
- Encode your prompts with BERT
- Display cosine similarity matrix for each prompt pair
- Generate and save images for each text description

***

## Example Images

### Futuristic Cityscape
[1]

### Bookstore Café
[2]

### Lakeside Cabin in Autumn
[3]

***

## File Structure

- `3_text_to_image_tokenizer.py`: Main script to run the embedding, similarity, and image generation pipeline.
- `image1-1.jpg`, `image2.jpg`, `image3.jpg`: Sample output images (see above).

***

## How It Works

1. **Text Embedding**: Prompts are passed through BERT to get [CLS] token embeddings.
2. **Similarity Metric**: Cosine similarity is used to show which prompts are thematically similar.
3. **Stable Diffusion Pipeline**: Prompts are sent through the Stable Diffusion model to generate images.
4. **Saving/Displaying Images**: Images are saved and optionally displayed.

***

## License

MIT

***

## Contact

Open an issue or pull request for improvements or feature requests.

***
