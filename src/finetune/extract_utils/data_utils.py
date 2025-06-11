# Hello, I'm Chung, AI Engineer in MISA JSC!
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from .config import (
    IGNORE_INDEX, FINAL_OTHERS_LABEL, 
    esg_label2id, e_sub_label2id, s_sub_label2id, g_sub_label2id,
    final_label2id, BATCH_SIZE
)

def load_jsonl(file_path):
    """
    Load data from a JSONL file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

class ESGMultiTaskDataset(Dataset):
    """
    Dataset class for ESG multi-task classification
    
    Attributes:
        data: List of dictionaries containing the data
        tokenizer: Tokenizer to use for encoding the text
        max_len: Maximum sequence length
    """
    def __init__(self, data, tokenizer, max_len=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])
        factor_label_str = item['factor']
        sub_factor_label_str = item['sub_factor']

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )

        if factor_label_str in esg_label2id:
            esg_label = esg_label2id[factor_label_str]
        else:
            esg_label = esg_label2id["Others_ESG"]

        e_sub_label = IGNORE_INDEX
        s_sub_label = IGNORE_INDEX
        g_sub_label = IGNORE_INDEX

        final_true_label_str = FINAL_OTHERS_LABEL
        if factor_label_str == "E" and sub_factor_label_str in e_sub_label2id:
            e_sub_label = e_sub_label2id[sub_factor_label_str]
            final_true_label_str = sub_factor_label_str
        elif factor_label_str == "S" and sub_factor_label_str in s_sub_label2id:
            s_sub_label = s_sub_label2id[sub_factor_label_str]
            final_true_label_str = sub_factor_label_str
        elif factor_label_str == "G" and sub_factor_label_str in g_sub_label2id:
            g_sub_label = g_sub_label2id[sub_factor_label_str]
            final_true_label_str = sub_factor_label_str

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'esg_label': esg_label,
            'e_sub_label': e_sub_label,
            's_sub_label': s_sub_label,
            'g_sub_label': g_sub_label,
            'final_true_label': final_label2id[final_true_label_str]
        }

def custom_collate_fn(batch_items, data_collator):
    """
    Custom collate function for the DataLoader
    
    Args:
        batch_items: Batch of items from the dataset
        data_collator: DataCollator to use for padding
        
    Returns:
        Dictionary containing the batched inputs and labels
    """
    processed_batch = data_collator([{k: v for k, v in item.items() 
                                    if k in ['input_ids', 'attention_mask']} 
                                   for item in batch_items])
    processed_batch['esg_labels'] = torch.tensor([item['esg_label'] for item in batch_items], dtype=torch.long)
    processed_batch['e_sub_labels'] = torch.tensor([item['e_sub_label'] for item in batch_items], dtype=torch.long)
    processed_batch['s_sub_labels'] = torch.tensor([item['s_sub_label'] for item in batch_items], dtype=torch.long)
    processed_batch['g_sub_labels'] = torch.tensor([item['g_sub_label'] for item in batch_items], dtype=torch.long)
    processed_batch['final_true_labels'] = torch.tensor([item['final_true_label'] for item in batch_items], dtype=torch.long)
    return processed_batch

def prepare_dataloaders(all_data, tokenizer, data_collator):
    """
    Prepare train and validation DataLoaders
    
    Args:
        all_data: List of dictionaries containing the data
        tokenizer: Tokenizer to use for encoding the text
        data_collator: DataCollator to use for padding
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        all_data, 
        test_size=0.1, 
        random_state=42, 
        # stratify=[d['factor'] for d in all_data]
    )
    
    # Create datasets
    train_dataset = ESGMultiTaskDataset(train_data, tokenizer)
    val_dataset = ESGMultiTaskDataset(val_data, tokenizer)
    
    # Create dataloaders with custom collate function
    collate_fn = lambda batch: custom_collate_fn(batch, data_collator)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader
