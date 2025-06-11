# Hello, I'm Chung, AI Engineer in MISA JSC!
import torch
import torch.optim as optim
from transformers import get_scheduler
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .config import (
    DEVICE, PHASE_1_EPOCHS, PHASE_2_EPOCHS,
    BEST_MODEL_SAVE_PATH
)
from .loss_def import calculate_combined_loss

def check_for_nan(tensor, name):
    """
    Check if a tensor contains any NaN values
    
    Args:
        tensor: Tensor to check
        name: Name of the tensor for logging
        
    Returns:
        Boolean indicating if NaN was found
    """
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def train_epoch(model, train_dataloader, optimizer, lr_scheduler, criterion, epoch, total_epochs, phase_name):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        optimizer: Optimizer to use
        lr_scheduler: Learning rate scheduler
        criterion: Loss function to use
        epoch: Current epoch number
        total_epochs: Total number of epochs for this phase
        phase_name: Name of the training phase (e.g., "Phase 1")
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    if phase_name == "Phase 1":
        model.gte_model.eval()  # Keep GTE in eval mode if frozen
    
    total_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Training {phase_name}]")
    
    for batch_idx, batch in enumerate(train_progress_bar):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        true_esg = batch['esg_labels'].to(DEVICE)
        true_e_sub = batch['e_sub_labels'].to(DEVICE)
        true_s_sub = batch['s_sub_labels'].to(DEVICE)
        true_g_sub = batch['g_sub_labels'].to(DEVICE)

        optimizer.zero_grad()
        
        try:
            # Forward pass
            logits_esg, logits_e_sub, logits_s_sub, logits_g_sub = model(input_ids, attention_mask)
            
            # Check for NaN in logits
            if (check_for_nan(logits_esg, "logits_esg") or 
                check_for_nan(logits_e_sub, "logits_e_sub") or 
                check_for_nan(logits_s_sub, "logits_s_sub") or 
                check_for_nan(logits_g_sub, "logits_g_sub")):
                print(f"NaN logits detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping batch.")
                continue
                
            # Calculate combined loss
            combined_loss = calculate_combined_loss(
                criterion, logits_esg, logits_e_sub, logits_s_sub, logits_g_sub,
                true_esg, true_e_sub, true_s_sub, true_g_sub
            )
            
            if torch.isnan(combined_loss):
                print(f"NaN in combined loss at epoch {epoch+1}, batch {batch_idx+1}")
                continue
                
            combined_loss.backward()
            
            # Gradient clipping
            clip_norm = 0.5 if phase_name == "Phase 1" else 0.1  # More aggressive clipping in Phase 2
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            
            optimizer.step()
            lr_scheduler.step()
            
            total_train_loss += combined_loss.item()
            train_progress_bar.set_postfix({'loss': combined_loss.item()})
            
        except RuntimeError as e:
            print(f"Error in training batch: {e}")
            continue

    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss

def validate(model, val_dataloader, criterion, epoch, total_epochs, phase_name):
    """
    Validate the model
    
    Args:
        model: Model to validate
        val_dataloader: DataLoader for validation data
        criterion: Loss function to use
        epoch: Current epoch number
        total_epochs: Total number of epochs for this phase
        phase_name: Name of the training phase (e.g., "Phase 1")
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    all_final_preds = []
    all_final_true = []
    total_val_loss = 0
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase_name} Validation]")

    with torch.no_grad():
        for batch in val_progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            true_esg_val = batch['esg_labels'].to(DEVICE)
            true_e_sub_val = batch['e_sub_labels'].to(DEVICE)
            true_s_sub_val = batch['s_sub_labels'].to(DEVICE)
            true_g_sub_val = batch['g_sub_labels'].to(DEVICE)
            final_true_labels_batch = batch['final_true_labels'].cpu().numpy()

            # Forward pass
            logits_esg, logits_e_sub, logits_s_sub, logits_g_sub = model(input_ids, attention_mask)
            
            # Calculate combined loss
            v_combined_loss = calculate_combined_loss(
                criterion, logits_esg, logits_e_sub, logits_s_sub, logits_g_sub,
                true_esg_val, true_e_sub_val, true_s_sub_val, true_g_sub_val
            )
            total_val_loss += v_combined_loss.item()

            # Get predictions for each level
            from .config import esg_id2label, e_sub_id2label, s_sub_id2label, g_sub_id2label, final_label2id

            prob_esg = torch.softmax(logits_esg, dim=1)
            pred_esg_indices = torch.argmax(prob_esg, dim=1)
            
            batch_final_preds_str = []
            for i in range(input_ids.size(0)):
                pred_esg_idx = pred_esg_indices[i].item()
                pred_esg_label_str = esg_id2label[pred_esg_idx]
                
                current_final_pred_label_str = "Others"

                if pred_esg_label_str == "E":
                    prob_e_sub_sample = torch.softmax(logits_e_sub[i], dim=0)
                    pred_e_sub_idx = torch.argmax(prob_e_sub_sample).item()
                    current_final_pred_label_str = e_sub_id2label[pred_e_sub_idx]
                elif pred_esg_label_str == "S":
                    prob_s_sub_sample = torch.softmax(logits_s_sub[i], dim=0)
                    pred_s_sub_idx = torch.argmax(prob_s_sub_sample).item()
                    current_final_pred_label_str = s_sub_id2label[pred_s_sub_idx]
                elif pred_esg_label_str == "G":
                    prob_g_sub_sample = torch.softmax(logits_g_sub[i], dim=0)
                    pred_g_sub_idx = torch.argmax(prob_g_sub_sample).item()
                    current_final_pred_label_str = g_sub_id2label[pred_g_sub_idx]
                
                batch_final_preds_str.append(current_final_pred_label_str)
            
            # Convert string predictions to ID predictions for accuracy calculation
            batch_final_preds_ids = [final_label2id[lbl_str] for lbl_str in batch_final_preds_str]
            
            all_final_preds.extend(batch_final_preds_ids)
            all_final_true.extend(final_true_labels_batch)

    # Calculate metrics
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(all_final_true, all_final_preds)
    
    # Calculate F1 scores for imbalanced data
    # Macro F1: Calculate F1 for each class independently, then average (treats all classes equally)
    # Better for imbalanced datasets as it gives equal importance to each class
    val_f1_macro = f1_score(all_final_true, all_final_preds, average='macro')
    
    # Weighted F1: Calculate F1 for each class, but weight by class frequency
    # Takes class imbalance into account - more frequent classes have more influence
    val_f1_weighted = f1_score(all_final_true, all_final_preds, average='weighted')
    
    # Calculate precision and recall (macro average)
    precision, recall, _, _ = precision_recall_fscore_support(
        all_final_true, all_final_preds, average='macro'
    )
    
    metrics = {
        "loss": avg_val_loss,
        "accuracy": val_accuracy,
        "f1_macro": val_f1_macro,
        "f1_weighted": val_f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall
    }
    
    return metrics

def train_model_phase1(model, train_dataloader, val_dataloader, criterion):
    """
    Phase 1 training: Train only classification heads
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        criterion: Loss function to use
        
    Returns:
        Best validation accuracy and model state
    """
    print("Phase 1: Training only classification heads")
    
    # Freeze GTE model
    for param in model.gte_model.parameters():
        param.requires_grad = False

    # Setup optimizer for non-frozen parts
    mlp_params = list(model.shared_projector.parameters()) + \
                list(model.esg_head.parameters()) + list(model.e_sub_head.parameters()) + \
                list(model.s_sub_head.parameters()) + list(model.g_sub_head.parameters())
    optimizer = optim.AdamW(mlp_params, lr=2e-4, weight_decay=0.01)

    # Setup learning rate scheduler
    num_training_steps = PHASE_1_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Initialize early stopping variables
    best_val_accuracy = 0.0
    patience = 3
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(PHASE_1_EPOCHS):
        # Train one epoch
        avg_train_loss = train_epoch(
            model, train_dataloader, optimizer, lr_scheduler, criterion, 
            epoch, PHASE_1_EPOCHS, "Phase 1"
        )
        
        print(f"\nPhase 1 Epoch {epoch+1}/{PHASE_1_EPOCHS} finished. Average Training Loss: {avg_train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, val_dataloader, criterion, epoch, PHASE_1_EPOCHS, "Phase 1")
        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]
        val_f1_macro = val_metrics["f1_macro"]
        val_f1_weighted = val_metrics["f1_weighted"]
        
        print(f"Phase 1 Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"F1 Scores - Macro: {val_f1_macro:.4f}, Weighted: {val_f1_weighted:.4f}")
        
        # Check for improvement using F1-macro score instead of accuracy for imbalanced data
        if val_f1_macro > best_val_accuracy:  # Reusing variable name but tracking F1 now
            best_val_accuracy = val_f1_macro
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, BEST_MODEL_SAVE_PATH)
            print(f"New best model saved with F1-macro score: {best_val_accuracy:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
                
    return best_val_accuracy, best_model_state

def train_model_phase2(model, train_dataloader, val_dataloader, criterion, best_val_accuracy_phase1):
    """
    Phase 2 training: Fine-tune entire model
    
    Args:
        model: Model to train (with best weights from Phase 1)
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        criterion: Loss function to use
        best_val_accuracy_phase1: Best validation accuracy from Phase 1
        
    Returns:
        Best validation accuracy and best model state
    """
    print("\nPhase 2: Fine-tuning the entire model")
    
    # Unfreeze GTE model
    for param in model.gte_model.parameters():
        param.requires_grad = True
    
    # Setup optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.gte_model.parameters(), 'lr': 1e-5},  # Very small LR for GTE
        {'params': model.shared_projector.parameters(), 'lr': 5e-5},
        {'params': model.esg_head.parameters(), 'lr': 5e-5},
        {'params': model.e_sub_head.parameters(), 'lr': 5e-5},
        {'params': model.s_sub_head.parameters(), 'lr': 5e-5},
        {'params': model.g_sub_head.parameters(), 'lr': 5e-5}
    ], weight_decay=0.01)
    
    # Setup learning rate scheduler
    num_training_steps = PHASE_2_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Initialize early stopping variables
    best_val_accuracy_phase2 = best_val_accuracy_phase1
    patience = 3
    patience_counter = 0
    best_model_state = model.state_dict().copy()
    
    # Training loop
    for epoch in range(PHASE_2_EPOCHS):
        # Train one epoch
        avg_train_loss = train_epoch(
            model, train_dataloader, optimizer, lr_scheduler, criterion, 
            epoch, PHASE_2_EPOCHS, "Phase 2"
        )
        
        print(f"\nPhase 2 Epoch {epoch+1}/{PHASE_2_EPOCHS} finished. Average Training Loss: {avg_train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, val_dataloader, criterion, epoch, PHASE_2_EPOCHS, "Phase 2")
        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]
        val_f1_macro = val_metrics["f1_macro"]
        val_f1_weighted = val_metrics["f1_weighted"]
        
        print(f"Phase 2 Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"F1 Scores - Macro: {val_f1_macro:.4f}, Weighted: {val_f1_weighted:.4f}")
        
        # Check for improvement using F1-macro score instead of accuracy for imbalanced data
        if val_f1_macro > best_val_accuracy_phase2:
            best_val_accuracy_phase2 = val_f1_macro
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, BEST_MODEL_SAVE_PATH)
            print(f"New best model saved with F1-macro score: {best_val_accuracy_phase2:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping in Phase 2 after {epoch+1} epochs")
                break
                
    return best_val_accuracy_phase2, best_model_state
