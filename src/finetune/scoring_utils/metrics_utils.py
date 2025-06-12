# Hello, I'm Chung, AI Engineer in MISA JSC!
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions
    
    Args:
        eval_pred: Tuple of (logits, labels) from the evaluation step
        
    Returns:
        Dictionary containing metrics (accuracy, f1, precision, recall)
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='weighted', 
        zero_division=0
    )
    
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
