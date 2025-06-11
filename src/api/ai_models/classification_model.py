import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple
from .config import config

class HierarchicalMTLClassifier(nn.Module):
    """
    Hierarchical Multi-Task Learning Classifier for ESG sub-factor classification.
    Copied exactly from finetune/extract.py to ensure accuracy.
    """
    
    def __init__(self, model_name, num_esg_classes, num_e_sub_classes,
                 num_s_sub_classes, num_g_sub_classes,
                 shared_hidden_dim_factor=0.5, head_hidden_dim_factor=0.25,
                 dropout_rate=0.1, freeze_gte=True):
        super(HierarchicalMTLClassifier, self).__init__()
        
        self.gte_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            cache_dir=config.classification_config['cache_dir']
        )
        embedding_dim = self.gte_model.config.hidden_size

        if freeze_gte:
            for param in self.gte_model.parameters():
                param.requires_grad = False
        
        shared_hidden_dim = int(embedding_dim * shared_hidden_dim_factor)
        head_hidden_dim = int(embedding_dim * head_hidden_dim_factor)

        # Added layer normalization for better numerical stability
        self.shared_projector = nn.Sequential(
            nn.Linear(embedding_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification heads with layer normalization
        self.esg_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_esg_classes)
        )
        
        self.e_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_e_sub_classes)
        )
        
        self.s_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_s_sub_classes)
        )
        
        self.g_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_g_sub_classes)
        )
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights using Xavier initialization"""
        for module in [self.shared_projector, self.esg_head, 
                      self.e_sub_head, self.s_sub_head, self.g_sub_head]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.gte_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Added clipping to prevent extreme values
        cls_embedding = torch.clamp(cls_embedding, min=-100, max=100)
        
        shared_features = self.shared_projector(cls_embedding)
        logits_esg = self.esg_head(shared_features)
        logits_e_sub = self.e_sub_head(shared_features)
        logits_s_sub = self.s_sub_head(shared_features)
        logits_g_sub = self.g_sub_head(shared_features)
        return logits_esg, logits_e_sub, logits_s_sub, logits_g_sub

class ClassificationModelManager:
    """Manager for sub-factor classification model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    async def load(self) -> None:
        """Load classification model with exact logic from app.py"""
        print("Loading sub-factor classification model and tokenizer...")
        try:
            cls_conf = config.classification_config
            arch_conf = cls_conf['architecture']
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                cls_conf['base_model'], 
                cache_dir=cls_conf['cache_dir'],
                trust_remote_code=config.loading_config['trust_remote_code']
            )
            
            # Initialize model architecture
            self.model = HierarchicalMTLClassifier(
                model_name=cls_conf['base_model'],
                num_esg_classes=config.num_esg_classes,
                num_e_sub_classes=config.num_e_sub_classes,
                num_s_sub_classes=config.num_s_sub_classes,
                num_g_sub_classes=config.num_g_sub_classes,
                shared_hidden_dim_factor=arch_conf['shared_hidden_dim_factor'],
                head_hidden_dim_factor=arch_conf['head_hidden_dim_factor'],
                dropout_rate=arch_conf['dropout_rate'],
                freeze_gte=arch_conf['freeze_gte']
            ).to(config.device)
            
            # Load weights
            self.model.load_state_dict(torch.load(cls_conf['weights_path'], map_location=config.device))
            print(f"Loaded sub-factor model weights from {cls_conf['weights_path']}")
                
            if config.loading_config['eval_mode']:
                self.model.eval()
            print("Sub-factor classification model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"Error loading sub-factor classification model: {e}")
            raise e
    
    def predict(self, text: str) -> Tuple[str, str]:
        """Classify text to predict ESG factor and sub-factor"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Classification model not loaded")
        
        cls_conf = config.classification_config
        
        # Tokenize input - exact logic from app.py
        tokenized_input = self.tokenizer(
            text,
            truncation=True,
            max_length=cls_conf['max_length'],
            return_tensors="pt"
        )
        
        # Move tensors to device
        input_ids = tokenized_input.input_ids.to(config.device)
        attention_mask = tokenized_input.attention_mask.to(config.device)
        
        # Get model predictions - exact logic from app.py
        with torch.no_grad():
            logits_esg, logits_e_sub, logits_s_sub, logits_g_sub = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # Get ESG factor prediction
            prob_esg = torch.softmax(logits_esg, dim=1)
            pred_esg_idx = torch.argmax(prob_esg, dim=1).item()
            pred_esg_label = config.esg_id2label[pred_esg_idx]
            
            # Default sub-factor to "Others"
            sub_factor_label = config.final_others_label
            
            # Get sub-factor prediction based on ESG factor
            if pred_esg_label == "E":
                prob_e_sub = torch.softmax(logits_e_sub[0], dim=0)
                pred_e_sub_idx = torch.argmax(prob_e_sub).item()
                sub_factor_label = config.e_sub_id2label[pred_e_sub_idx]
            elif pred_esg_label == "S":
                prob_s_sub = torch.softmax(logits_s_sub[0], dim=0)
                pred_s_sub_idx = torch.argmax(prob_s_sub).item()
                sub_factor_label = config.s_sub_id2label[pred_s_sub_idx]
            elif pred_esg_label == "G":
                prob_g_sub = torch.softmax(logits_g_sub[0], dim=0)
                pred_g_sub_idx = torch.argmax(prob_g_sub).item()
                sub_factor_label = config.g_sub_id2label[pred_g_sub_idx]
        
        return pred_esg_label, sub_factor_label
