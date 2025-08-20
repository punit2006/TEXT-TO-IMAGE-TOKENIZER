from tokenizer_pipeline.embeddings import preprocess_text
from tokenizer_pipeline.similarity import compute_cosine_similarity
from tokenizer_pipeline.diffusion_generator import generate_image_from_text
from tokenizer_pipeline.utils import save_images

if __name__ == "__main__":
    # Step 1: Text descriptions
    text_descriptions = [
        "A beautiful sunset over the mountains.",
        "A bustling city street filled with people.",
        "A quiet forest with a clear blue sky."
    ]

    # Step 2: Generate embeddings
    embeddings = preprocess_text(text_descriptions)
    print("Embeddings shape:", embeddings.shape)

    # Step 3: Compute similarity
    similarity_matrix = compute_cosine_similarity(embeddings)
    print("Cosine Similarity Matrix:")
    print(similarity_matrix)

    # Step 4: Stable Diffusion text-to-image
    prompts = [
        "A serene lakeside cabin surrounded by autumn foliage.",
        "A futuristic cityscape at night with neon lights and flying cars.",
        "A cozy bookstore café with wooden shelves filled with books and a warm fireplace."
    ]
    images = generate_image_from_text(prompts)

    # Step 5: Save images
    filenames = ["image1.png", "image2.png", "image3.png"]
    save_images(images, filenames)
    print("Images saved successfully.")
