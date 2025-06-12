# Hello, I'm Chung, AI Engineer in MISA JSC!
import torch
import torch.nn as nn
from .config import IGNORE_INDEX, LOSS_WEIGHTS

class SafeCrossEntropyLoss(nn.Module):
    """
    A safer implementation of CrossEntropyLoss that handles the IGNORE_INDEX better
    by properly masking and averaging only over valid targets.
    
    This helps prevent NaN issues during training when a batch might not have
    any valid targets for a specific task.
    """
    def __init__(self, ignore_index=-100):
        super(SafeCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
    def forward(self, logits, targets):
        """
        Calculate loss, ensuring proper handling of ignored indices
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            
        Returns:
            Scalar loss value (or zero tensor with gradient if no valid targets)
        """
        # Calculate standard CE loss
        loss = self.ce_loss(logits, targets)
        
        # Create mask for valid (non-ignored) targets
        valid_mask = (targets != self.ignore_index).float()
        
        # Count number of valid targets
        num_valid = torch.sum(valid_mask)
        
        # If no valid targets, return zero loss
        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Return average loss over valid targets
        return torch.sum(loss * valid_mask) / num_valid

def calculate_combined_loss(criterion, logits_esg, logits_e_sub, logits_s_sub, 
                            logits_g_sub, true_esg, true_e_sub, true_s_sub, true_g_sub):
    """
    Calculate the combined weighted loss from all classification tasks
    
    Args:
        criterion: Loss function to use
        logits_esg: Logits for ESG classification
        logits_e_sub: Logits for E subcategories
        logits_s_sub: Logits for S subcategories
        logits_g_sub: Logits for G subcategories
        true_esg: Ground truth ESG labels
        true_e_sub: Ground truth E subcategory labels
        true_s_sub: Ground truth S subcategory labels
        true_g_sub: Ground truth G subcategory labels
        
    Returns:
        Combined weighted loss value
    """
    # Calculate individual losses
    loss_esg_val = criterion(logits_esg, true_esg)
    loss_e_sub_val = criterion(logits_e_sub, true_e_sub)
    loss_s_sub_val = criterion(logits_s_sub, true_s_sub)
    loss_g_sub_val = criterion(logits_g_sub, true_g_sub)
    
    # Check for NaN in individual losses
    has_nan = (torch.isnan(loss_esg_val) or torch.isnan(loss_e_sub_val) or 
               torch.isnan(loss_s_sub_val) or torch.isnan(loss_g_sub_val))
    
    if has_nan:
        # Log information for debugging
        print(f"NaN detected in losses: ESG: {loss_esg_val.item()}, E_sub: {loss_e_sub_val.item()}, " 
              f"S_sub: {loss_s_sub_val.item()}, G_sub: {loss_g_sub_val.item()}")
        
        # Replace NaN values with zeros to continue training
        loss_esg_val = torch.nan_to_num(loss_esg_val, nan=0.0)
        loss_e_sub_val = torch.nan_to_num(loss_e_sub_val, nan=0.0)
        loss_s_sub_val = torch.nan_to_num(loss_s_sub_val, nan=0.0)
        loss_g_sub_val = torch.nan_to_num(loss_g_sub_val, nan=0.0)
    
    # Compute the combined weighted loss
    combined_loss = (LOSS_WEIGHTS["esg"] * loss_esg_val +
                     LOSS_WEIGHTS["e_sub"] * loss_e_sub_val +
                     LOSS_WEIGHTS["s_sub"] * loss_s_sub_val +
                     LOSS_WEIGHTS["g_sub"] * loss_g_sub_val)
    
    return combined_loss
