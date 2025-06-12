# Hello, I'm Chung, AI Engineer in MISA JSC!
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch

from .config import (
    DATA_PATH, LABEL2ID, TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE
)

def load_data(data_file=DATA_PATH):
    """
    Load data from a JSONL file and prepare texts and labels
    
    Args:
        data_file: Path to the JSONL file
        
    Returns:
        Tuple of (texts, labels_str, labels)
    """
    texts = []
    labels_str = []
    
    # Load data from JSONL file
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            labels_str.append(record["factor"])
    
    # Convert string labels to IDs
    labels = [LABEL2ID[label] for label in labels_str]
    
    return texts, labels_str, labels

def create_dataset_dict(data_file=DATA_PATH):
    """
    Create a Hugging Face DatasetDict with train and validation splits
    
    Args:
        data_file: Path to the JSONL file
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    # Load and prepare data
    texts, labels_str, labels = load_data(data_file)
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, 
        test_size=TRAIN_TEST_SPLIT_RATIO, 
        random_state=RANDOM_STATE, 
        stratify=labels  # Maintain class distribution
    )
    
    # Create Hugging Face Datasets
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    
    # Return DatasetDict
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
