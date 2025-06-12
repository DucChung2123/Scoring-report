"""
Script to upload scoring model to Hugging Face Hub
"""
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi
from api.ai_models.config import config

def create_scoring_model_card(repo_name: str) -> str:
    """Create model card for the scoring model repository"""
    return f"""---
language: 
- en
- vi
tags:
- esg
- scoring
- classification
- sustainability
datasets:
- custom
library_name: transformers
pipeline_tag: text-classification
---

# ESG Scoring Model

This model performs ESG (Environmental, Social, Governance) scoring for text classification.

## Model Description

- **Model Type**: Sequence Classification for ESG Scoring
- **Language**: English, Vietnamese
- **Task**: ESG Factor Scoring (E, S, G)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Example usage
text = "The company has implemented renewable energy solutions to reduce carbon emissions."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)

# Get probabilities for each ESG factor
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    
# Get scores for each factor
e_score = probabilities[0, 0].item()  # Environmental score
s_score = probabilities[0, 1].item()  # Social score 
g_score = probabilities[0, 2].item()  # Governance score

print(f"Environmental: {{e_score:.4f}}")
print(f"Social: {{s_score:.4f}}")
print(f"Governance: {{g_score:.4f}}")
```

## Training Details

- **Training Data**: Custom ESG dataset
- **Training Approach**: Fine-tuned for ESG factor scoring
- **Labels**: E (Environmental), S (Social), G (Governance)

## Model Performance

The model achieves strong performance on ESG scoring tasks across multiple languages.

## Limitations

- Trained primarily on English and Vietnamese text
- Performance may vary on domain-specific or technical content
- Best performance on texts similar to training data distribution

## Citation

If you use this model, please cite:

```bibtex
@misc{{esg_scoring_model,
  title={{ESG Scoring Model}},
  author={{Chung}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{repo_name}}}
}}
```
"""

def upload_scoring_model():
    """Upload scoring model to HuggingFace Hub"""
    print("🚀 Starting ESG Scoring Model Upload Process...")
    
    # Configuration
    scoring_conf = config.scoring_config
    MODEL_PATH = scoring_conf['path']
    
    # You need to set these
    REPO_NAME = input("Enter your Hugging Face repo name for scoring model (format: username/model-name): ").strip()
    if not REPO_NAME or '/' not in REPO_NAME:
        print("❌ Invalid repo name. Format should be: username/model-name")
        return
    
    PRIVATE = input("Make repository private? (y/N): ").strip().lower() == 'y'
    
    try:
        print(f"\n📦 Loading scoring model from: {MODEL_PATH}")
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=len(scoring_conf['labels']),
            id2label={v: k for k, v in scoring_conf['labels'].items()},
            label2id=scoring_conf['labels'],
            trust_remote_code=config.loading_config['trust_remote_code']
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=config.loading_config['trust_remote_code']
        )
        
        # Test the model
        print("🧪 Testing scoring model...")
        test_text = "The company has implemented renewable energy solutions to reduce carbon emissions."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=scoring_conf['max_length'])
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        for factor, idx in scoring_conf['labels'].items():
            score = probabilities[0, idx].item()
            print(f"  {factor}: {score:.4f}")
        
        print("✅ Model test successful!")
        
        # Save to temporary directory
        temp_dir = "./temp_scoring_upload"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            print("💾 Saving model and tokenizer...")
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Create model card
            model_card_content = create_scoring_model_card(REPO_NAME)
            with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(model_card_content)
            
            print("🚀 Uploading to Hugging Face Hub...")
            
            # Upload to hub
            api = HfApi()
            api.create_repo(REPO_NAME, private=PRIVATE, exist_ok=True)
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=REPO_NAME,
                repo_type="model"
            )
            
            print(f"✅ Successfully uploaded to https://huggingface.co/{REPO_NAME}")
            
            print("\n💡 Usage example:")
            print(f"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("{REPO_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")

text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    
e_score = probabilities[0, 0].item()
s_score = probabilities[0, 1].item() 
g_score = probabilities[0, 2].item()
            """)
            
        finally:
            # Clean up temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"❌ Error during upload process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    upload_scoring_model()
