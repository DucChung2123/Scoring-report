# Hello, I'm Chung, AI Engineer in MISA JSC!
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm

# Import components from our modular structure
from extract_utils.config import MODEL_NAME, DEVICE, DATA_PATH, PHASE_2_EPOCHS
from extract_utils.data_utils import load_jsonl, prepare_dataloaders
from extract_utils.model_def import create_model
from extract_utils.loss_def import SafeCrossEntropyLoss
from extract_utils.train_utils import train_model_phase1, train_model_phase2

def main():
    """Main function to run the ESG multi-task fine-tuning"""
    
    # Enable anomaly detection for better NaN debugging
    torch.autograd.set_detect_anomaly(True)
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    all_data = load_jsonl(DATA_PATH)
    print(f"Loaded {len(all_data)} examples from {DATA_PATH}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir="models",
        trust_remote_code=True,
    )
    
    # Setup data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    # Prepare dataloaders
    train_dataloader, val_dataloader = prepare_dataloaders(all_data, tokenizer, data_collator)
    print(f"Train dataloader has {len(train_dataloader)} batches")
    print(f"Validation dataloader has {len(val_dataloader)} batches")
    
    # Initialize model
    model = create_model(MODEL_NAME, DEVICE, freeze_gte=True)
    
    # Initialize loss function
    criterion = SafeCrossEntropyLoss()
    
    # Phase 1: Train only the classification heads
    best_val_accuracy_phase1, best_model_state_phase1 = train_model_phase1(
        model, train_dataloader, val_dataloader, criterion
    )
    
    # Load best model from Phase 1
    print("Loading best model from Phase 1")
    model.load_state_dict(best_model_state_phase1)
    
    # Phase 2: Fine-tune the entire model
    if PHASE_2_EPOCHS > 0:
        best_val_accuracy_phase2, best_model_state_phase2 = train_model_phase2(
            model, train_dataloader, val_dataloader, criterion, best_val_accuracy_phase1
        )
        
        # Load the best model (which could be from either phase)
        print(f"\nTraining completed.")
        print(f"Best validation F1-macro score from Phase 1: {best_val_accuracy_phase1:.4f}")
        print(f"Best validation F1-macro score from Phase 2: {best_val_accuracy_phase2:.4f}")
        print(f"Final best F1-macro score: {max(best_val_accuracy_phase1, best_val_accuracy_phase2):.4f}")
        
        # Load the best model from saved checkpoint
        from extract_utils.config import BEST_MODEL_SAVE_PATH
        model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))
    else:
        print(f"\nTraining completed.")
        print(f"Using Phase 1 model as final model (F1-macro score: {best_val_accuracy_phase1:.4f})")
    
    # Final model is ready for inference
    model.eval()
    print("Model is ready for inference.")

if __name__ == "__main__":
    main()
