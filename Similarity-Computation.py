from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(embeddings):
    """Compute cosine similarity matrix for embeddings"""
    embeddings_array = embeddings.detach().numpy()
    similarity_matrix = cosine_similarity(embeddings_array)
    return similarity_matrix
