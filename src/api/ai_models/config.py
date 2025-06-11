import torch
from typing import Dict, Any, List
from src.api.settings import settings

class ModelConfig:
    """
    Model configuration manager using settings from YAML files.
    Clean and config-driven approach following best practices.
    """
    
    def __init__(self):
        self.model_conf = settings.MODEL_CONF
        self._setup_device()
        self._setup_labels()
    
    def _setup_device(self) -> None:
        """Setup device based on config"""
        device_conf = self.model_conf['device']
        if device_conf['auto_detect']:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_conf['fallback'])
        print(f"Using device: {self.device}")
    
    def _setup_labels(self) -> None:
        """Setup all label mappings from config"""
        esg_conf = self.model_conf['esg_categories']
        
        # Main ESG categories
        self.esg_categories = esg_conf['main']
        self.esg_label2id = {label: i for i, label in enumerate(self.esg_categories)}
        self.esg_id2label = {i: label for i, label in enumerate(self.esg_categories)}
        
        # Sub-factors
        sub_factors = esg_conf['sub_factors']
        self.sub_factors_e = sub_factors['E']
        self.sub_factors_s = sub_factors['S']
        self.sub_factors_g = sub_factors['G']
        
        # Create sub-factor mappings
        self.e_sub_label2id = {label: i for i, label in enumerate(self.sub_factors_e)}
        self.e_sub_id2label = {i: label for i, label in enumerate(self.sub_factors_e)}
        
        self.s_sub_label2id = {label: i for i, label in enumerate(self.sub_factors_s)}
        self.s_sub_id2label = {i: label for i, label in enumerate(self.sub_factors_s)}
        
        self.g_sub_label2id = {label: i for i, label in enumerate(self.sub_factors_g)}
        self.g_sub_id2label = {i: label for i, label in enumerate(self.sub_factors_g)}
        
        # Final combined mappings
        all_sub_factors = self.sub_factors_e + self.sub_factors_s + self.sub_factors_g
        self.final_others_label = esg_conf['final_others_label']
        all_final_labels = all_sub_factors + [self.final_others_label]
        self.final_label2id = {label: i for i, label in enumerate(all_final_labels)}
        self.final_id2label = {i: label for i, label in enumerate(all_final_labels)}
        
        # Constants
        self.ignore_index = esg_conf['ignore_index']
    
    @property
    def scoring_config(self) -> Dict[str, Any]:
        """Get scoring model configuration"""
        return self.model_conf['scoring_model']
    
    @property
    def classification_config(self) -> Dict[str, Any]:
        """Get classification model configuration"""
        return self.model_conf['classification_model']
    
    @property
    def loading_config(self) -> Dict[str, Any]:
        """Get model loading configuration"""
        return self.model_conf['loading']
    
    # Properties for easy access to counts
    @property
    def num_esg_classes(self) -> int:
        return len(self.esg_categories)
    
    @property
    def num_e_sub_classes(self) -> int:
        return len(self.sub_factors_e)
    
    @property
    def num_s_sub_classes(self) -> int:
        return len(self.sub_factors_s)
    
    @property
    def num_g_sub_classes(self) -> int:
        return len(self.sub_factors_g)

# Global config instance
config = ModelConfig()
