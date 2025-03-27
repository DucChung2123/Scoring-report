from src.services.search.embedding import Embedding
from src.services.search.sentence_search import SentenceSearch


class SearchEngine:
    def __init__(self, 
                 sentences: list[str]):
        Embedding.initialize()
        self.texts = sentences
        self.embeddings, _ = Embedding.get_embeddings(texts=sentences)
        self.search_instance = SentenceSearch(self.embeddings, sentences)
        
    def search(self, 
               query: str, 
               k: int = 5,
               threshold: float = None):
        query_embedding, _ = Embedding.get_embeddings(texts=[query])
        results = self.search_instance.search(query_embedding, k)
        if threshold:
            results = [(text, score) for text, score in results if score > threshold]
        return results

# if __name__ == "__main__":
#     sentences = [
#         "Hôm nay trời nắng to",
#         "Hôm nay trời mưa",
#         "Trường học màu xanh",
#         "Tôi đi học về",
#         "Bão đang về hôm nay, trời sẽ có giông"
#     ]
#     SearchEngine_instance = SearchEngine(sentences)
#     query = "Thời tiết"
#     results = SearchEngine_instance.search(query, k=3, threshold=0.5)
#     print(results)