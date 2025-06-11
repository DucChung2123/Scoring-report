# Hello, I'm Chung, AI Engineer in MISA JSC!
from transformers import AutoTokenizer, DataCollatorWithPadding

from .config import MODEL_NAME, MODEL_MAX_LENGTH, CACHE_DIR

def get_tokenizer():
    """
    Initialize and return the tokenizer
    
    Returns:
        Initialized tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    return tokenizer

def get_data_collator(tokenizer):
    """
    Create a data collator for dynamic padding
    
    Args:
        tokenizer: Tokenizer to use
        
    Returns:
        DataCollatorWithPadding instance
    """
    return DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(examples, tokenizer=None):
    """
    Tokenize text examples
    
    Args:
        examples: Batch of examples containing 'text' field
        tokenizer: Optional tokenizer (if None, will be initialized)
        
    Returns:
        Tokenized examples
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MODEL_MAX_LENGTH
    )

def prepare_tokenized_datasets(raw_datasets):
    """
    Tokenize datasets and remove the 'text' column
    
    Args:
        raw_datasets: DatasetDict with 'train' and 'validation' splits
        
    Returns:
        DatasetDict with tokenized data
    """
    tokenizer = get_tokenizer()
    
    # Tokenize datasets
    try:
        tokenized_datasets = raw_datasets.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True
        )
    except Exception as e:
        print(f"Error during tokenization: {e}")
        raise
    
    # Remove 'text' column as it's not needed after tokenization
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    
    return tokenized_datasets, tokenizer
