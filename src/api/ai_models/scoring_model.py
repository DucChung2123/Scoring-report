import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import config

class ScoringModelManager:
    """Manager for ESG scoring model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    async def load(self) -> None:
        """Load scoring model - supports both HuggingFace Hub and local checkpoint"""
        print("Loading scoring model and tokenizer...")
        try:
            score_conf = config.scoring_config
            
            # Check if hub_path is configured for HuggingFace Hub loading
            if 'hub_path' in score_conf and score_conf['hub_path']:
                print(f"Using HuggingFace Hub loading from: {score_conf['hub_path']}")
                await self._load_from_hub(score_conf)
            else:
                print("Using local checkpoint loading")
                await self._load_from_checkpoint(score_conf)
                
            print("Scoring model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"Error loading scoring model: {e}")
            raise e
    
    async def _load_from_hub(self, score_conf: dict) -> None:
        """Load model from HuggingFace Hub (simple loading)"""
        # Get device for this model
        model_device = config.get_model_device(score_conf)
        
        # Load model from Hub with cache
        self.model = AutoModelForSequenceClassification.from_pretrained(
            score_conf['hub_path'],
            trust_remote_code=config.loading_config['trust_remote_code'],
            cache_dir=score_conf.get('cache_dir', 'models')
        )
        
        # Load tokenizer with cache
        self.tokenizer = AutoTokenizer.from_pretrained(
            score_conf['hub_path'],
            trust_remote_code=config.loading_config['trust_remote_code'],
            cache_dir=score_conf.get('cache_dir', 'models')
        )
        
        # Move to specified device and set eval mode
        if config.loading_config['device_placement']:
            self.model.to(model_device)
            print(f"Moved scoring model to device: {model_device}")
        
        if config.loading_config['eval_mode']:
            self.model.eval()
            
        print(f"Loaded model from HuggingFace Hub: {score_conf['hub_path']}")
    
    async def _load_from_checkpoint(self, score_conf: dict) -> None:
        """Load model from local checkpoint (original logic)"""
        # Get device for this model
        model_device = config.get_model_device(score_conf)
        
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
            self.model.to(model_device)
            print(f"Moved scoring model to device: {model_device}")
        
        if config.loading_config['eval_mode']:
            self.model.eval()
    
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
        
        # Move tensors to model device
        model_device = config.get_model_device(score_conf)
        input_ids = tokenized_input.input_ids.to(model_device)
        attention_mask = tokenized_input.attention_mask.to(model_device)
        
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
