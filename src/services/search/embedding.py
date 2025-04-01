from src.core.config import settings
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

class Embedding:
    model_name = settings.MODEL_NAME  # Should be 'Alibaba-NLP/gte-multilingual-base'
    model_dir = settings.MODEL_DIR
    batch_size = settings.BATCH_SIZE
    embedding_dimension = 768
    model = None
    tokenizer = None
    
    @classmethod
    def initialize(cls):
        """
        Khởi tạo model và tokenizer cho class
        """
        if cls.model is None or cls.tokenizer is None:
            cls._load_model()
        
    @classmethod
    def _load_model(cls):
        """
        Load model from directory. If not present, download and save it.
        Use GPU if available, otherwise use CPU.
        """
        # Check if model exists in model_dir
        os.makedirs(cls.model_dir, exist_ok=True)
        model_path = os.path.join(cls.model_dir, cls.model_name.split('/')[-1])
        
        try:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                cls.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                cls.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            else:
                print(f"Model not found in {model_path}. Downloading...")
                cls.model = AutoModel.from_pretrained(cls.model_name, trust_remote_code=True)
                cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, trust_remote_code=True)
                
                # Save the model
                cls.model.save_pretrained(model_path)
                cls.tokenizer.save_pretrained(model_path)
                print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        cls.embedding_dimension = cls.model.config.hidden_size
        cls.model = cls.model.to(device)
        cls.model.eval()
    
    @staticmethod
    def embeddings(texts, model, tokenizer, batch_size=32, embedding_dimension=768):
        """
        Generate embeddings for a list of texts using batching.
        Specifically for GTE multilingual models.
        
        Args:
            texts (list): List of texts to embed
            model: The embedding model
            tokenizer: The tokenizer
            batch_size (int): Batch size for processing
            embedding_dimension (int): The dimension of the output embeddings
            
        Returns:
            tuple: (embeddings, original_texts)
        """
        device = next(model.parameters()).device
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                # Tokenize
                batch_dict = tokenizer(
                    batch_texts, 
                    max_length=2048,
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                ).to(device)
                
                # Get model outputs
                outputs = model(**batch_dict)
                
                # Get the CLS token embedding (first token) and take the specified dimension
                batch_embeddings = outputs.last_hidden_state[:, 0][:embedding_dimension]
                
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            # Convert to numpy and store
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        return embeddings, texts
    
    @classmethod
    def get_embeddings(cls, texts):
        """
        Public method to get embeddings for a list of texts
        
        Args:
            texts (list): List of texts to embed
        
        Returns:
            tuple: (embeddings, original_texts)
        """
        # Kiểm tra xem model đã được khởi tạo chưa
        if cls.model is None or cls.tokenizer is None:
            cls.initialize()
            
        embeddings, original_texts = cls.embeddings(
            texts, 
            cls.model, 
            cls.tokenizer, 
            cls.batch_size, 
            cls.embedding_dimension
        )
        return embeddings, original_texts

# if __name__ == "__main__":
#     # Test the Embedding class
#     texts = [
#         "Việc xả thải trái phép gây ôi nhiễm không gian sống của sinh vật",
#         "Làm cách nào để tải một file trên youtobe?",
#     ]
#     Embedding.initialize()
#     embeddings, origin_text = Embedding.get_embeddings(texts)
#     query = "Tài nguyên, môi trường"
#     query_embedding, _ = Embedding.get_embeddings([query])
#     # score simmarity
#     scores = np.dot(embeddings, query_embedding.T)
#     print(scores)

