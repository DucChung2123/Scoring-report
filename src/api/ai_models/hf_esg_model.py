"""
Hugging Face compatible ESG Hierarchical Multi-Task Learning Model
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Dict, Any

class ESGHierarchicalConfig(PretrainedConfig):
    """
    Configuration class for ESG Hierarchical Multi-Task Learning model
    """
    model_type = "esg_hierarchical"
    
    def __init__(
        self,
        backbone_model_name: str = "Alibaba-NLP/gte-multilingual-base",
        num_esg_classes: int = 4,
        num_e_sub_classes: int = 3,
        num_s_sub_classes: int = 7,
        num_g_sub_classes: int = 5,
        shared_hidden_dim_factor: float = 0.5,
        head_hidden_dim_factor: float = 0.25,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = True,
        esg_categories: list = None,
        sub_factors_e: list = None,
        sub_factors_s: list = None,
        sub_factors_g: list = None,
        final_others_label: str = "Others",
        max_length: int = 4096,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Model architecture parameters
        self.backbone_model_name = backbone_model_name
        self.num_esg_classes = num_esg_classes
        self.num_e_sub_classes = num_e_sub_classes
        self.num_s_sub_classes = num_s_sub_classes
        self.num_g_sub_classes = num_g_sub_classes
        self.shared_hidden_dim_factor = shared_hidden_dim_factor
        self.head_hidden_dim_factor = head_hidden_dim_factor
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length
        
        # Label mappings
        self.esg_categories = esg_categories or ["E", "S", "G", "Others_ESG"]
        self.sub_factors_e = sub_factors_e or ["Emission", "Resource Use", "Product Innovation"]
        self.sub_factors_s = sub_factors_s or ["Community", "Diversity", "Employment", "HS", "HR", "PR", "Training"]
        self.sub_factors_g = sub_factors_g or ["BFunction", "BStructure", "Compensation", "Shareholder", "Vision"]
        self.final_others_label = final_others_label
        
        # Create label mappings
        self.esg_label2id = {label: i for i, label in enumerate(self.esg_categories)}
        self.esg_id2label = {i: label for i, label in enumerate(self.esg_categories)}
        
        self.e_sub_label2id = {label: i for i, label in enumerate(self.sub_factors_e)}
        self.e_sub_id2label = {i: label for i, label in enumerate(self.sub_factors_e)}
        
        self.s_sub_label2id = {label: i for i, label in enumerate(self.sub_factors_s)}
        self.s_sub_id2label = {i: label for i, label in enumerate(self.sub_factors_s)}
        
        self.g_sub_label2id = {label: i for i, label in enumerate(self.sub_factors_g)}
        self.g_sub_id2label = {i: label for i, label in enumerate(self.sub_factors_g)}
        
        # Final combined mappings
        all_sub_factors = self.sub_factors_e + self.sub_factors_s + self.sub_factors_g
        all_final_labels = all_sub_factors + [self.final_others_label]
        self.final_label2id = {label: i for i, label in enumerate(all_final_labels)}
        self.final_id2label = {i: label for i, label in enumerate(all_final_labels)}


class ESGHierarchicalForSequenceClassification(PreTrainedModel):
    """
    ESG Hierarchical Multi-Task Learning Model for Hugging Face Hub
    
    This model performs hierarchical ESG classification:
    1. First predicts main ESG category (E, S, G, Others_ESG)
    2. Then predicts corresponding sub-factor based on the main category
    """
    config_class = ESGHierarchicalConfig
    
    def __init__(self, config: ESGHierarchicalConfig, backbone=None):
        super().__init__(config)
        self.config = config
        
        # Use provided backbone or create empty one (weights will be loaded by from_pretrained)
        if backbone is not None:
            self.gte_model = backbone
        else:
            # During from_pretrained, just create the architecture - weights loaded automatically
            from transformers import AutoConfig
            backbone_config = AutoConfig.from_pretrained(config.backbone_model_name, trust_remote_code=True)
            self.gte_model = AutoModel.from_config(backbone_config, trust_remote_code=True)
        
        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.gte_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from backbone
        embedding_dim = self.gte_model.config.hidden_size
        shared_hidden_dim = int(embedding_dim * config.shared_hidden_dim_factor)
        head_hidden_dim = int(embedding_dim * config.head_hidden_dim_factor)
        
        # Shared projector layer
        self.shared_projector = nn.Sequential(
            nn.Linear(embedding_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Classification heads
        self.esg_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(config.dropout_rate),
            nn.Linear(head_hidden_dim, config.num_esg_classes)
        )
        
        self.e_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(config.dropout_rate),
            nn.Linear(head_hidden_dim, config.num_e_sub_classes)
        )
        
        self.s_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(config.dropout_rate),
            nn.Linear(head_hidden_dim, config.num_s_sub_classes)
        )
        
        self.g_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), 
            nn.Dropout(config.dropout_rate),
            nn.Linear(head_hidden_dim, config.num_g_sub_classes)
        )
        
        # Initialize weights
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
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        """
        Forward pass of the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding tokens
            labels: Dictionary containing target labels for each task
            return_dict: Whether to return dictionary format
            
        Returns:
            SequenceClassifierOutput containing logits and loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get backbone outputs
        backbone_outputs = self.gte_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Use CLS token embedding
        cls_embedding = backbone_outputs.last_hidden_state[:, 0, :]
        
        # Ensure dtype consistency - convert to float32
        cls_embedding = cls_embedding.float()
        
        # Apply clipping to prevent extreme values
        cls_embedding = torch.clamp(cls_embedding, min=-100, max=100)
        
        # Get shared features
        shared_features = self.shared_projector(cls_embedding)
        
        # Get logits from all heads
        esg_logits = self.esg_head(shared_features)
        e_sub_logits = self.e_sub_head(shared_features)
        s_sub_logits = self.s_sub_head(shared_features)
        g_sub_logits = self.g_sub_head(shared_features)
        
        # Prepare output
        logits = {
            'esg': esg_logits,
            'e_sub': e_sub_logits,
            's_sub': s_sub_logits,
            'g_sub': g_sub_logits
        }
        
        loss = None
        if labels is not None:
            # Calculate loss if labels provided
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            losses = []
            loss_weights = {"esg": 1.0, "e_sub": 0.7, "s_sub": 0.7, "g_sub": 0.7}
            
            for task_name, task_logits in logits.items():
                if task_name in labels:
                    task_loss = loss_fct(task_logits, labels[task_name])
                    losses.append(loss_weights[task_name] * task_loss)
            
            if losses:
                loss = sum(losses)
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[str, str]:
        """
        Hierarchical prediction method compatible with original implementation
        
        Returns:
            Tuple of (main_esg_factor, sub_factor)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get ESG factor prediction
            prob_esg = torch.softmax(logits['esg'], dim=1)
            pred_esg_idx = torch.argmax(prob_esg, dim=1).item()
            pred_esg_label = self.config.esg_id2label[pred_esg_idx]
            
            # Default sub-factor to "Others"
            sub_factor_label = self.config.final_others_label
            
            # Get sub-factor prediction based on ESG factor
            if pred_esg_label == "E":
                prob_e_sub = torch.softmax(logits['e_sub'][0], dim=0)
                pred_e_sub_idx = torch.argmax(prob_e_sub).item()
                sub_factor_label = self.config.e_sub_id2label[pred_e_sub_idx]
            elif pred_esg_label == "S":
                prob_s_sub = torch.softmax(logits['s_sub'][0], dim=0)
                pred_s_sub_idx = torch.argmax(prob_s_sub).item()
                sub_factor_label = self.config.s_sub_id2label[pred_s_sub_idx]
            elif pred_esg_label == "G":
                prob_g_sub = torch.softmax(logits['g_sub'][0], dim=0)
                pred_g_sub_idx = torch.argmax(prob_g_sub).item()
                sub_factor_label = self.config.g_sub_id2label[pred_g_sub_idx]
        
        return pred_esg_label, sub_factor_label


# Register the model
ESGHierarchicalConfig.register_for_auto_class()
ESGHierarchicalForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
