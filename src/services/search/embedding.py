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
    embedding_dimension = settings.EMBEDDING_DIMENSION if hasattr(settings, 'EMBEDDING_DIMENSION') else 768
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """
        Load model from directory. If not present, download and save it.
        Use GPU if available, otherwise use CPU.
        """
        # Check if model exists in model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, self.model_name.split('/')[-1])
        
        try:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            else:
                print(f"Model not found in {model_path}. Downloading...")
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                
                # Save the model
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model = self.model.to(device)
        self.model.eval()
    
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
                    max_length=8192,
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
    
    def get_embeddings(self, texts):
        """
        Public method to get embeddings for a list of texts
        
        Args:
            texts (list): List of texts to embed
        
        Returns:
            tuple: (embeddings, original_texts)
        """
        embeddings, original_texts = self.embeddings(
            texts, 
            self.model, 
            self.tokenizer, 
            self.batch_size, 
            self.embedding_dimension
        )
        return embeddings, original_texts


# if __name__ == "__main__":
#     # Test the Embedding class
#     texts = [
#         "Việc xả thải trái phép gây ôi nhiễm không gian sống của sinh vật",
#         "Làm cách nào để tải một file trên youtobe?",
#     ]
#     emb = Embedding()
#     embeddings, origin_text = emb.get_embeddings(texts)
#     query = "Tài nguyên, môi trường"
#     query_embedding, _ = emb.get_embeddings([query])
#     # score simmarity
#     scores = np.dot(embeddings, query_embedding.T)
#     print(scores)