# Hello, I'm Chung, AI Engineer in MISA JSC!
import torch
import numpy as np

# Import utilities from our modular structure
from scoring_utils.config import DEVICE
from scoring_utils.data_utils import create_dataset_dict
from scoring_utils.tokenization_utils import prepare_tokenized_datasets, get_data_collator
from scoring_utils.model_setup import create_model
from scoring_utils.metrics_utils import compute_metrics
from scoring_utils.trainer_setup import create_trainer

def main():
    """Main function to train an ESG classifier"""
    
    print(f"Using device: {DEVICE}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load and prepare datasets
    print("Loading and preparing datasets...")
    raw_datasets = create_dataset_dict()
    print(f"Created dataset dictionary with {len(raw_datasets['train'])} training examples "
          f"and {len(raw_datasets['validation'])} validation examples")
    
    # 2. Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets, tokenizer = prepare_tokenized_datasets(raw_datasets)
    print("Tokenization complete")
    
    # 3. Create data collator for dynamic padding
    data_collator = get_data_collator(tokenizer)
    
    # 4. Initialize model
    print("Initializing model...")
    model = create_model()
    
    # 5. Create trainer
    print("Setting up trainer...")
    trainer = create_trainer(
        model, 
        tokenizer, 
        tokenized_datasets, 
        data_collator, 
        compute_metrics
    )
    
    # 6. Train model
    print("Starting training...")
    trainer.train()
    
    # 7. Evaluate model
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    print("\nTraining completed successfully!")
    
if __name__ == "__main__":
    main()
