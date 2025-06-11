# load data
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
data_file = "data/processed/ESG_data.jsonl"
texts = []
labels_str = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        texts.append(record["text"])
        labels_str.append(record["factor"])

# Định nghĩa ánh xạ nhãn
label2id = {"E": 0, "S": 1, "G": 2}
id2label = {0: "E", 1: "S", 2: "G"}
num_labels = len(label2id)

labels = [label2id[label] for label in labels_str]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels # stratify để giữ tỷ lệ nhãn
)

# Tạo Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

# Gộp thành DatasetDict (nếu cần)
raw_datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# --- 2. Tokenization ---
from transformers import AutoTokenizer, DataCollatorWithPadding
model_name = "Alibaba-NLP/gte-multilingual-base" # Tên model trên Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)

MODEL_MAX_LENGTH = 8192

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MODEL_MAX_LENGTH
    )
    
try:
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
except Exception as e:
    print(f"Lỗi trong quá trình tokenization: {e}")
    exit()
    

# Xóa cột "text" vì không cần thiết sau tokenization
tokenized_datasets = tokenized_datasets.remove_columns(["text"])


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "Alibaba-NLP/gte-multilingual-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
    cache_dir="models",
    ignore_mismatched_sizes=True
)

import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding # Sẽ được Trainer sử dụng cho dynamic padding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nSử dụng thiết bị: {device}")


output_dir = "./results_gte_classifier_dynamic"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,                     # Số epochs, có thể cần điều chỉnh
    per_device_train_batch_size=16,          # Batch size, giảm nếu gặp lỗi OOM
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,                       # Tỷ lệ số bước warmup
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=max(1, len(tokenized_datasets["train"]) // (8 * 10)), # Log khoảng 10 lần/epoch
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",             # Chọn model tốt nhất dựa trên F1 score
    greater_is_better=True,
    # fp16=torch.cuda.is_available(),       # Bật mixed precision nếu có GPU và thư viện hỗ trợ (có thể cần `pip install accelerate`)
    report_to="none"                        # Tắt WandB/TensorBoard reporting nếu không cần
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,                    # Quan trọng: để lưu tokenizer cùng model
    data_collator=data_collator,            # Sử dụng DataCollator cho dynamic padding
    compute_metrics=compute_metrics,
)
trainer.train()