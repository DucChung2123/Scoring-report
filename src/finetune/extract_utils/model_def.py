# Hello, I'm Chung, AI Engineer in MISA JSC!
import torch
import torch.nn as nn
from transformers import AutoModel
from .config import (
    NUM_ESG_CLASSES, NUM_E_SUB_CLASSES, 
    NUM_S_SUB_CLASSES, NUM_G_SUB_CLASSES
)

class HierarchicalMTLClassifier(nn.Module):
    """
    Hierarchical Multi-Task Learning Classifier for ESG classification
    
    This model uses a pretrained language model as the base encoder,
    followed by a shared projector and multiple classification heads
    for different ESG-related tasks.
    
    Attributes:
        gte_model: The base encoder model
        shared_projector: Shared feature extractor after the base encoder
        esg_head: Classification head for ESG categories (E, S, G)
        e_sub_head: Classification head for E subcategories
        s_sub_head: Classification head for S subcategories
        g_sub_head: Classification head for G subcategories
    """
    def __init__(self, model_name, num_esg_classes, num_e_sub_classes,
                 num_s_sub_classes, num_g_sub_classes,
                 shared_hidden_dim_factor=0.5, head_hidden_dim_factor=0.25,
                 dropout_rate=0.1, freeze_gte=True):
        super(HierarchicalMTLClassifier, self).__init__()
        self.gte_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, cache_dir="models"
        )
        embedding_dim = self.gte_model.config.hidden_size

        if freeze_gte:
            for param in self.gte_model.parameters():
                param.requires_grad = False
        
        shared_hidden_dim = int(embedding_dim * shared_hidden_dim_factor)
        head_hidden_dim = int(embedding_dim * head_hidden_dim_factor)

        # Shared projector with layer normalization for better numerical stability
        self.shared_projector = nn.Sequential(
            nn.Linear(embedding_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ESG classification head
        self.esg_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_esg_classes)
        )
        
        # E subcategory classification head
        self.e_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_e_sub_classes)
        )
        
        # S subcategory classification head
        self.s_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_s_sub_classes)
        )
        
        # G subcategory classification head
        self.g_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_g_sub_classes)
        )
        
        # Initialize weights using Xavier initialization
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
        """
        Forward pass through the model
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask for the input tokens
            
        Returns:
            Tuple of logits for ESG, E subcategories, S subcategories, and G subcategories
        """
        outputs = self.gte_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Clip to prevent extreme values
        cls_embedding = torch.clamp(cls_embedding, min=-100, max=100)
        
        shared_features = self.shared_projector(cls_embedding)
        logits_esg = self.esg_head(shared_features)
        logits_e_sub = self.e_sub_head(shared_features)
        logits_s_sub = self.s_sub_head(shared_features)
        logits_g_sub = self.g_sub_head(shared_features)
        
        return logits_esg, logits_e_sub, logits_s_sub, logits_g_sub

def create_model(model_name, device, freeze_gte=True):
    """
    Create and initialize the model
    
    Args:
        model_name: Name of the pretrained model to use
        device: Device to place the model on
        freeze_gte: Whether to freeze the base encoder weights
        
    Returns:
        Initialized model
    """
    model = HierarchicalMTLClassifier(
        model_name=model_name,
        num_esg_classes=NUM_ESG_CLASSES,
        num_e_sub_classes=NUM_E_SUB_CLASSES,
        num_s_sub_classes=NUM_S_SUB_CLASSES,
        num_g_sub_classes=NUM_G_SUB_CLASSES,
        freeze_gte=freeze_gte
    ).to(device)
    
    return model
