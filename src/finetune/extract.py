import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import numpy as np
from functools import partial

# Enable anomaly detection for better NaN debugging
torch.autograd.set_detect_anomaly(True)

# --- Label definitions and mappings (unchanged) ---
SUB_FACTORS_E = ["Emission", "Resource Use", "Product Innovation"]
SUB_FACTORS_S = ["Community", "Diversity", "Employment", "HS", "HR", "PR", "Training"]
SUB_FACTORS_G = ["BFunction", "BStructure", "Compensation", "Shareholder", "Vision"]
ALL_SUB_FACTORS_ORDERED = SUB_FACTORS_E + SUB_FACTORS_S + SUB_FACTORS_G

ESG_CATEGORIES = ["E", "S", "G", "Others_ESG"]

esg_label2id = {label: i for i, label in enumerate(ESG_CATEGORIES)}
esg_id2label = {i: label for i, label in enumerate(ESG_CATEGORIES)}
NUM_ESG_CLASSES = len(ESG_CATEGORIES)

e_sub_label2id = {label: i for i, label in enumerate(SUB_FACTORS_E)}
e_sub_id2label = {i: label for i, label in enumerate(SUB_FACTORS_E)}
NUM_E_SUB_CLASSES = len(SUB_FACTORS_E)

s_sub_label2id = {label: i for i, label in enumerate(SUB_FACTORS_S)}
s_sub_id2label = {i: label for i, label in enumerate(SUB_FACTORS_S)}
NUM_S_SUB_CLASSES = len(SUB_FACTORS_S)

g_sub_label2id = {label: i for i, label in enumerate(SUB_FACTORS_G)}
g_sub_id2label = {i: label for i, label in enumerate(SUB_FACTORS_G)}
NUM_G_SUB_CLASSES = len(SUB_FACTORS_G)

FINAL_OTHERS_LABEL = "Others"
ALL_FINAL_LABELS = ALL_SUB_FACTORS_ORDERED + [FINAL_OTHERS_LABEL]
final_label2id = {label: i for i, label in enumerate(ALL_FINAL_LABELS)}
final_id2label = {i: label for i, label in enumerate(ALL_FINAL_LABELS)}

IGNORE_INDEX = -100
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Load data (unchanged) ---
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

file_path = "data/processed/ESG_data.jsonl"
all_data = load_jsonl(file_path)

# --- Dataset Class (unchanged) ---
class ESGMultiTaskDataset(Dataset):
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

# --- Tokenizer (unchanged) ---
tokenizer = AutoTokenizer.from_pretrained(
    "Alibaba-NLP/gte-multilingual-base",
    cache_dir="models",
    trust_remote_code=True,
)

# --- Model Class (with added layer normalization and dropout adjustments) ---
class HierarchicalMTLClassifier(nn.Module):
    def __init__(self, model_name, num_esg_classes, num_e_sub_classes,
                 num_s_sub_classes, num_g_sub_classes,
                 shared_hidden_dim_factor=0.5, head_hidden_dim_factor=0.25,
                 dropout_rate=0.1, freeze_gte=True):  # Reduced dropout to 0.1
        super(HierarchicalMTLClassifier, self).__init__()
        self.gte_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir="models")
        embedding_dim = self.gte_model.config.hidden_size

        if freeze_gte:
            for param in self.gte_model.parameters():
                param.requires_grad = False
        
        shared_hidden_dim = int(embedding_dim * shared_hidden_dim_factor)
        head_hidden_dim = int(embedding_dim * head_hidden_dim_factor)

        # Added layer normalization for better numerical stability
        self.shared_projector = nn.Sequential(
            nn.Linear(embedding_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),  # Added LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Added layer normalization in each head
        self.esg_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),  # Added LayerNorm
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_esg_classes)
        )
        
        self.e_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),  # Added LayerNorm
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_e_sub_classes)
        )
        
        self.s_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),  # Added LayerNorm
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_s_sub_classes)
        )
        
        self.g_sub_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, head_hidden_dim), 
            nn.LayerNorm(head_hidden_dim),  # Added LayerNorm
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_g_sub_classes)
        )
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights using Xavier initialization"""
        for module in [self.shared_projector, self.esg_head, 
                      self.e_sub_head, self.s_sub_head, self.g_sub_head]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.gte_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Added clipping to prevent extreme values
        cls_embedding = torch.clamp(cls_embedding, min=-100, max=100)
        
        shared_features = self.shared_projector(cls_embedding)
        logits_esg = self.esg_head(shared_features)
        logits_e_sub = self.e_sub_head(shared_features)
        logits_s_sub = self.s_sub_head(shared_features)
        logits_g_sub = self.g_sub_head(shared_features)
        return logits_esg, logits_e_sub, logits_s_sub, logits_g_sub

# --- Initialize Model (with freeze_gte set to True initially) ---
model = HierarchicalMTLClassifier(
    model_name=MODEL_NAME,
    num_esg_classes=NUM_ESG_CLASSES,
    num_e_sub_classes=NUM_E_SUB_CLASSES,
    num_s_sub_classes=NUM_S_SUB_CLASSES,
    num_g_sub_classes=NUM_G_SUB_CLASSES,
    freeze_gte=True  # Start with frozen GTE model
).to(DEVICE)

# --- DataLoaders (unchanged) ---
train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42, stratify=[d['factor'] for d in all_data])

train_dataset = ESGMultiTaskDataset(train_data, tokenizer)
val_dataset = ESGMultiTaskDataset(val_data, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def custom_collate_fn(batch_items):
    processed_batch = data_collator([{k: v for k, v in item.items() if k in ['input_ids', 'attention_mask']} for item in batch_items])
    processed_batch['esg_labels'] = torch.tensor([item['esg_label'] for item in batch_items], dtype=torch.long)
    processed_batch['e_sub_labels'] = torch.tensor([item['e_sub_label'] for item in batch_items], dtype=torch.long)
    processed_batch['s_sub_labels'] = torch.tensor([item['s_sub_label'] for item in batch_items], dtype=torch.long)
    processed_batch['g_sub_labels'] = torch.tensor([item['g_sub_label'] for item in batch_items], dtype=torch.long)
    processed_batch['final_true_labels'] = torch.tensor([item['final_true_label'] for item in batch_items], dtype=torch.long)
    return processed_batch

BATCH_SIZE = 8
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

# --- Loss, Optimizer, Scheduler (with improved loss function) ---
# Custom loss function that handles IGNORE_INDEX better
class SafeCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(SafeCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
    def forward(self, logits, targets):
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

criterion = SafeCrossEntropyLoss(ignore_index=IGNORE_INDEX)

# Two-phase training strategy
PHASE_1_EPOCHS = 3
PHASE_2_EPOCHS = 6
TOTAL_EPOCHS = PHASE_1_EPOCHS + PHASE_2_EPOCHS

# Phase 1: Train only the classification heads
print("Phase 1: Training only classification heads")
for param in model.gte_model.parameters():
    param.requires_grad = False

mlp_params = list(model.shared_projector.parameters()) + \
             list(model.esg_head.parameters()) + list(model.e_sub_head.parameters()) + \
             list(model.s_sub_head.parameters()) + list(model.g_sub_head.parameters())
optimizer = optim.AdamW(mlp_params, lr=2e-4, weight_decay=0.01)  # Added weight decay

loss_weights = {"esg": 1.0, "e_sub": 0.7, "s_sub": 0.7, "g_sub": 0.7}
num_training_steps = PHASE_1_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

# --- Training Loop with more NaN handling and early stopping ---
model.to(DEVICE)

# Variables for early stopping
best_val_accuracy = 0.0
patience = 3
patience_counter = 0
BEST_MODEL_SAVE_PATH = "output/best_hierarchical_mtl_4096_esg_model_state.pth"

# Function to check for NaN and log it
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# Training loop for Phase 1
for epoch in range(PHASE_1_EPOCHS):
    model.train()
    model.gte_model.eval()  # Keep GTE in eval mode since it's frozen

    total_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{PHASE_1_EPOCHS} [Training Phase 1]")
    
    for batch_idx, batch in enumerate(train_progress_bar):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        true_esg = batch['esg_labels'].to(DEVICE)
        true_e_sub = batch['e_sub_labels'].to(DEVICE)
        true_s_sub = batch['s_sub_labels'].to(DEVICE)
        true_g_sub = batch['g_sub_labels'].to(DEVICE)

        optimizer.zero_grad()
        
        # Forward pass with gradient monitoring
        logits_esg, logits_e_sub, logits_s_sub, logits_g_sub = model(input_ids, attention_mask)
        
        # Check for NaN in logits
        if (check_for_nan(logits_esg, "logits_esg") or 
            check_for_nan(logits_e_sub, "logits_e_sub") or 
            check_for_nan(logits_s_sub, "logits_s_sub") or 
            check_for_nan(logits_g_sub, "logits_g_sub")):
            print(f"NaN logits detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping batch.")
            continue
            
        # Calculate loss with safer implementation
        try:
            loss_esg_val = criterion(logits_esg, true_esg)
            loss_e_sub_val = criterion(logits_e_sub, true_e_sub)
            loss_s_sub_val = criterion(logits_s_sub, true_s_sub)
            loss_g_sub_val = criterion(logits_g_sub, true_g_sub)
            
            # Safety check for NaN in individual losses
            if (torch.isnan(loss_esg_val) or torch.isnan(loss_e_sub_val) or 
                torch.isnan(loss_s_sub_val) or torch.isnan(loss_g_sub_val)):
                print(f"NaN in individual loss at epoch {epoch+1}, batch {batch_idx+1}")
                print(f"Loss ESG: {loss_esg_val.item()}, E_sub: {loss_e_sub_val.item()}, "
                      f"S_sub: {loss_s_sub_val.item()}, G_sub: {loss_g_sub_val.item()}")
                # Skip this batch
                continue
                
            combined_loss = (loss_weights["esg"] * loss_esg_val +
                             loss_weights["e_sub"] * loss_e_sub_val +
                             loss_weights["s_sub"] * loss_s_sub_val +
                             loss_weights["g_sub"] * loss_g_sub_val)
            
            if torch.isnan(combined_loss):
                print(f"NaN in combined loss at epoch {epoch+1}, batch {batch_idx+1}")
                continue
                
            combined_loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            lr_scheduler.step()
            
            total_train_loss += combined_loss.item()
            train_progress_bar.set_postfix({'loss': combined_loss.item()})
            
        except RuntimeError as e:
            print(f"Error in training batch: {e}")
            continue

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"\nEpoch {epoch+1}/{PHASE_1_EPOCHS} finished. Average Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    all_final_preds = []
    all_final_true = []
    total_val_loss = 0
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{PHASE_1_EPOCHS} [Validation]")

    with torch.no_grad():
        for batch in val_progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            true_esg_val = batch['esg_labels'].to(DEVICE)
            true_e_sub_val = batch['e_sub_labels'].to(DEVICE)
            true_s_sub_val = batch['s_sub_labels'].to(DEVICE)
            true_g_sub_val = batch['g_sub_labels'].to(DEVICE)
            final_true_labels_batch = batch['final_true_labels'].cpu().numpy()

            logits_esg, logits_e_sub, logits_s_sub, logits_g_sub = model(input_ids, attention_mask)
            
            # Validation loss
            v_loss_esg = criterion(logits_esg, true_esg_val)
            v_loss_e = criterion(logits_e_sub, true_e_sub_val)
            v_loss_s = criterion(logits_s_sub, true_s_sub_val)
            v_loss_g = criterion(logits_g_sub, true_g_sub_val)
            v_combined_loss = (loss_weights["esg"] * v_loss_esg + loss_weights["e_sub"] * v_loss_e +
                               loss_weights["s_sub"] * v_loss_s + loss_weights["g_sub"] * v_loss_g)
            total_val_loss += v_combined_loss.item()

            # Get final predictions
            prob_esg = torch.softmax(logits_esg, dim=1)
            pred_esg_indices = torch.argmax(prob_esg, dim=1)
            
            batch_final_preds_str = []
            for i in range(input_ids.size(0)):
                pred_esg_idx = pred_esg_indices[i].item()
                pred_esg_label_str = esg_id2label[pred_esg_idx]
                
                current_final_pred_label_str = FINAL_OTHERS_LABEL

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
            
            batch_final_preds_ids = [final_label2id[lbl_str] for lbl_str in batch_final_preds_str]
            
            all_final_preds.extend(batch_final_preds_ids)
            all_final_true.extend(final_true_labels_batch)

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(all_final_true, all_final_preds)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Final Label Accuracy: {val_accuracy:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

# Load best model from Phase 1
print("Loading best model from Phase 1")
model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))

# Phase 2: Fine-tune the entire model with a very small learning rate
if PHASE_2_EPOCHS > 0:
    print("\nPhase 2: Fine-tuning the entire model")
    # Unfreeze GTE model with a very small learning rate
    for param in model.gte_model.parameters():
        param.requires_grad = True
    
    # Use a much smaller learning rate for GTE
    optimizer = optim.AdamW([
        {'params': model.gte_model.parameters(), 'lr': 1e-6},  # Very small LR for GTE
        {'params': model.shared_projector.parameters(), 'lr': 5e-5},
        {'params': model.esg_head.parameters(), 'lr': 5e-5},
        {'params': model.e_sub_head.parameters(), 'lr': 5e-5},
        {'params': model.s_sub_head.parameters(), 'lr': 5e-5},
        {'params': model.g_sub_head.parameters(), 'lr': 5e-5}
    ], weight_decay=0.01)
    
    num_training_steps = PHASE_2_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Reset early stopping variables
    best_val_accuracy_phase2 = best_val_accuracy
    patience_counter = 0
    
    # Training loop for Phase 2
    for epoch in range(PHASE_2_EPOCHS):
        model.train()
        
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{PHASE_2_EPOCHS} [Training Phase 2]")
        
        for batch_idx, batch in enumerate(train_progress_bar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            true_esg = batch['esg_labels'].to(DEVICE)
            true_e_sub = batch['e_sub_labels'].to(DEVICE)
            true_s_sub = batch['s_sub_labels'].to(DEVICE)
            true_g_sub = batch['g_sub_labels'].to(DEVICE)

            optimizer.zero_grad()
            
            try:
                logits_esg, logits_e_sub, logits_s_sub, logits_g_sub = model(input_ids, attention_mask)
                
                # Check for NaN in logits
                if (check_for_nan(logits_esg, "logits_esg") or 
                    check_for_nan(logits_e_sub, "logits_e_sub") or 
                    check_for_nan(logits_s_sub, "logits_s_sub") or 
                    check_for_nan(logits_g_sub, "logits_g_sub")):
                    print(f"NaN logits detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping batch.")
                    continue
                
                loss_esg_val = criterion(logits_esg, true_esg)
                loss_e_sub_val = criterion(logits_e_sub, true_e_sub)
                loss_s_sub_val = criterion(logits_s_sub, true_s_sub)
                loss_g_sub_val = criterion(logits_g_sub, true_g_sub)
                
                combined_loss = (loss_weights["esg"] * loss_esg_val +
                                 loss_weights["e_sub"] * loss_e_sub_val +
                                 loss_weights["s_sub"] * loss_s_sub_val +
                                 loss_weights["g_sub"] * loss_g_sub_val)
                
                if torch.isnan(combined_loss):
                    print(f"NaN loss detected at phase 2, epoch {epoch+1}, batch {batch_idx+1}")
                    continue
                    
                combined_loss.backward()
                
                # More aggressive gradient clipping in phase 2
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()
                lr_scheduler.step()
                
                total_train_loss += combined_loss.item()
                train_progress_bar.set_postfix({'loss': combined_loss.item()})
                
            except RuntimeError as e:
                print(f"Error in training batch: {e}")
                continue
                
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"\nPhase 2 Epoch {epoch+1}/{PHASE_2_EPOCHS} finished. Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation in Phase 2
        model.eval()
        all_final_preds = []
        all_final_true = []
        total_val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Phase 2 Epoch {epoch+1}/{PHASE_2_EPOCHS} [Validation]")

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                true_esg_val = batch['esg_labels'].to(DEVICE)
                true_e_sub_val = batch['e_sub_labels'].to(DEVICE)
                true_s_sub_val = batch['s_sub_labels'].to(DEVICE)
                true_g_sub_val = batch['g_sub_labels'].to(DEVICE)
                final_true_labels_batch = batch['final_true_labels'].cpu().numpy()

                logits_esg, logits_e_sub, logits_s_sub, logits_g_sub = model(input_ids, attention_mask)
                
                # Validation loss
                v_loss_esg = criterion(logits_esg, true_esg_val)
                v_loss_e = criterion(logits_e_sub, true_e_sub_val)
                v_loss_s = criterion(logits_s_sub, true_s_sub_val)
                v_loss_g = criterion(logits_g_sub, true_g_sub_val)
                v_combined_loss = (loss_weights["esg"] * v_loss_esg + loss_weights["e_sub"] * v_loss_e +
                                  loss_weights["s_sub"] * v_loss_s + loss_weights["g_sub"] * v_loss_g)
                total_val_loss += v_combined_loss.item()

                # Get final predictions
                prob_esg = torch.softmax(logits_esg, dim=1)
                pred_esg_indices = torch.argmax(prob_esg, dim=1)
                
                batch_final_preds_str = []
                for i in range(input_ids.size(0)):
                    pred_esg_idx = pred_esg_indices[i].item()
                    pred_esg_label_str = esg_id2label[pred_esg_idx]
                    
                    current_final_pred_label_str = FINAL_OTHERS_LABEL

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
                
                batch_final_preds_ids = [final_label2id[lbl_str] for lbl_str in batch_final_preds_str]
                
                all_final_preds.extend(batch_final_preds_ids)
                all_final_true.extend(final_true_labels_batch)
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = accuracy_score(all_final_true, all_final_preds)
            print(f"Phase 2 Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            # Save best model from Phase 2
            if val_accuracy > best_val_accuracy_phase2:
                best_val_accuracy_phase2 = val_accuracy
                torch.save(model.state_dict(), "best_phase2_" + BEST_MODEL_SAVE_PATH)
                print(f"New best Phase 2 model saved with accuracy: {best_val_accuracy_phase2:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping in Phase 2 after {epoch+1} epochs")
                    break

print("Training completed.")
print(f"Best validation accuracy from Phase 1: {best_val_accuracy:.4f}")
if PHASE_2_EPOCHS > 0:
    print(f"Best validation accuracy from Phase 2: {best_val_accuracy_phase2:.4f}")
    
    # Determine which model to use as final
    if best_val_accuracy_phase2 > best_val_accuracy:
        print(f"Using Phase 2 model as final model (accuracy: {best_val_accuracy_phase2:.4f})")
        best_model_path = "best_phase2_" + BEST_MODEL_SAVE_PATH
    else:
        print(f"Using Phase 1 model as final model (accuracy: {best_val_accuracy:.4f})")
        best_model_path = BEST_MODEL_SAVE_PATH
        
    # Load the best model for inference
    model.load_state_dict(torch.load(best_model_path))
else:
    print(f"Using Phase 1 model as final model (accuracy: {best_val_accuracy:.4f})")
    
# Final model is now ready for inference
model.eval()