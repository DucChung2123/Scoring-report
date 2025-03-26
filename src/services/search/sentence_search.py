import faiss
import numpy as np

class SentenceSearch:
    
    def __init__(self, embeddings, texts):
        # Ensure embeddings is numpy array and has correct data type for FAISS
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.texts = texts
        self.index = self._build_index()
    
    def _build_index(self):
        """
        Build a FAISS index for the embeddings
        """
        dimension = self.embeddings.shape[1]
        print(f"Building FAISS index with {len(self.texts)} documents, dimension {dimension}")
        
        # Use Inner Product for normalized vectors (cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        index.add(self.embeddings)
        return index

    def search(self, query_embedding, k=5):
        """
        Search for the most similar sentences to the query
        
        Args:
            query_embedding: Embedding vector of query text
            k (int): The number of results to return
        
        Returns:
            list: A list of the top k most similar sentences with scores
        """
        # Make sure query_embedding is a 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert to float32 if needed
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, min(k, len(self.texts)))
        
        results = []
        for i in range(indices.shape[1]):  
            idx = indices[0, i]  
            score = scores[0, i]  
            
            if idx >= 0 and idx < len(self.texts):  
                results.append((self.texts[idx], float(score)))
        
        return results

if __name__ == "__main__":
    from src.services.search.embedding import Embedding
    emb = Embedding()
    texts_list = ["Hôm nay trời nắng to", "Hôm nay trời mưa", "Trường học màu xanh", "Tôi đi học về"]
    embeddings, texts = emb.get_embeddings(texts_list)
    
    search = SentenceSearch(embeddings, texts)
    
    query = "Thời tiết"
    
    query_embedding, _ = emb.get_embeddings([query])
    
    results = search.search(query_embedding, k=2)
    
    for result in results:
        print(result)