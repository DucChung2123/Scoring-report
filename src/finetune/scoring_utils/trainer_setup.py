# Hello, I'm Chung, AI Engineer in MISA JSC!
from transformers import TrainingArguments, Trainer

from .config import (
    OUTPUT_DIR, NUM_TRAIN_EPOCHS, BATCH_SIZE, 
    WARMUP_RATIO, WEIGHT_DECAY
)

def get_training_args():
    """
    Create and return TrainingArguments with optimal settings
    
    Returns:
        TrainingArguments object
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,                
        per_device_train_batch_size=BATCH_SIZE,          
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,                   
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        # Log approximately 10 times per epoch
        logging_steps=10,  
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",             
        greater_is_better=True,
        # Disable reporting to save resources
        report_to="none"                        
    )
    
    return training_args

def create_trainer(model, tokenizer, tokenized_datasets, data_collator, compute_metrics):
    """
    Create and return a Trainer
    
    Args:
        model: Model to train
        tokenizer: Tokenizer to use
        tokenized_datasets: DatasetDict with tokenized data
        data_collator: Data collator for dynamic padding
        compute_metrics: Function to compute metrics during evaluation
        
    Returns:
        Trainer object
    """
    training_args = get_training_args()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,                    
        data_collator=data_collator,            
        compute_metrics=compute_metrics,
    )
    
    return trainer
