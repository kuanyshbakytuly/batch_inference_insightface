from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_sim(embeddings_frame_array, embeddings_array):
    similarity = cosine_similarity(embeddings_frame_array, embeddings_array)
    indexes = np.argmax(similarity, axis=1)
    similarities = similarity[np.arange(len(indexes)), indexes]
    return np.column_stack((indexes, similarities))