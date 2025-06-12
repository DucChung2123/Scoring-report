# Hello, I'm Chung, AI Engineer in MISA JSC!
from transformers import AutoModelForSequenceClassification

from .config import (
    MODEL_NAME, CACHE_DIR, NUM_LABELS, ID2LABEL, LABEL2ID, DEVICE
)

def create_model():
    """
    Create and initialize a model for sequence classification
    
    Returns:
        Initialized model placed on the appropriate device
    """
    # Load model from Hugging Face
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        ignore_mismatched_sizes=True  # Allow for different classifier head sizes
    )
    
    # Move model to appropriate device (GPU if available)
    model = model.to(DEVICE)
    
    print(f"\nModel loaded and moved to device: {DEVICE}")
    
    return model
