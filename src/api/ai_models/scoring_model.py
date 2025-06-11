import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import config

class ScoringModelManager:
    """Manager for ESG scoring model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    async def load(self) -> None:
        """Load scoring model with exact logic from app.py"""
        print("Loading scoring model and tokenizer...")
        try:
            score_conf = config.scoring_config
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                score_conf['path'], 
                trust_remote_code=config.loading_config['trust_remote_code']
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                score_conf['path'],
                num_labels=len(score_conf['labels']),
                id2label={v: k for k, v in score_conf['labels'].items()},
                label2id=score_conf['labels'],
                trust_remote_code=config.loading_config['trust_remote_code']
            )
            
            if config.loading_config['device_placement']:
                self.model.to(config.device)
            
            if config.loading_config['eval_mode']:
                self.model.eval()
                
            print("Scoring model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"Error loading scoring model: {e}")
            raise e
    
    def predict(self, text: str, factor: str) -> float:
        """Score text for given ESG factor"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Scoring model not loaded")
        
        score_conf = config.scoring_config
        
        # Validate factor - exact logic from app.py
        factor = factor.upper()
        if factor not in score_conf['labels']:
            valid_factors = ", ".join(score_conf['labels'].keys())
            raise ValueError(f"Invalid factor: {factor}. Must be one of: {valid_factors}")
        
        # Tokenize input - exact logic from app.py
        tokenized_input = self.tokenizer(
            text,
            truncation=True,
            max_length=score_conf['max_length'],
            return_tensors="pt"
        )
        
        # Move tensors to device
        input_ids = tokenized_input.input_ids.to(config.device)
        attention_mask = tokenized_input.attention_mask.to(config.device)
        
        # Get model predictions - exact logic from app.py
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get the probability for the specified factor
            factor_idx = score_conf['labels'][factor]
            factor_probability = probabilities[0, factor_idx].item()
        
        return factor_probability
