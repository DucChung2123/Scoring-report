"""
Script to convert existing model weights and upload to Hugging Face Hub
"""
import torch
import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi
from api.ai_models.hf_esg_model import ESGHierarchicalConfig, ESGHierarchicalForSequenceClassification
from api.ai_models.config import config

def create_model_card(repo_name: str) -> str:
    """Create model card for the repository"""
    return f"""---
language: 
- en
- vi
tags:
- esg
- classification
- hierarchical
- multi-task-learning
- sustainability
datasets:
- custom
library_name: transformers
pipeline_tag: text-classification
---

# ESG Hierarchical Multi-Task Learning Model

This model performs hierarchical ESG (Environmental, Social, Governance) classification using a multi-task learning approach.

## Model Description

- **Model Type**: Hierarchical Multi-Task Classifier
- **Backbone**: Alibaba-NLP/gte-multilingual-base  
- **Language**: English, Vietnamese
- **Task**: ESG Factor and Sub-factor Classification

## Architecture

The model uses a hierarchical approach:
1. **Main ESG Classification**: Predicts E, S, G, or Others_ESG
2. **Sub-factor Classification**: Based on main category, predicts specific sub-factors:
   - **E (Environmental)**: Emission, Resource Use, Product Innovation
   - **S (Social)**: Community, Diversity, Employment, HS, HR, PR, Training  
   - **G (Governance)**: BFunction, BStructure, Compensation, Shareholder, Vision

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("{repo_name}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base")

# Example usage
text = "The company has implemented renewable energy solutions to reduce carbon emissions."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)

# Get predictions
esg_factor, sub_factor = model.predict(inputs.input_ids, inputs.attention_mask)
print(f"ESG Factor: {{esg_factor}}, Sub-factor: {{sub_factor}}")
```

## Training Details

- **Training Data**: Custom ESG dataset
- **Training Approach**: Two-phase training (freeze backbone → fine-tune entire model)
- **Loss Function**: Weighted multi-task loss
- **Optimization**: AdamW with learning rate scheduling

## Model Performance

The model achieves strong performance on ESG classification tasks with hierarchical prediction accuracy.

## Limitations

- Trained primarily on English and Vietnamese text
- Performance may vary on domain-specific or technical ESG content
- Best performance on texts similar to training data distribution

## Citation

If you use this model, please cite:

```bibtex
@misc{{esg_hierarchical_model,
  title={{ESG Hierarchical Multi-Task Learning Model}},
  author={{Chung}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{repo_name}}}
}}
```
"""

def load_and_convert_model(weights_path: str, backbone_model_name: str):
    """
    Load existing PyTorch weights and convert to HF format
    """
    print("Loading backbone model...")
    # Load the backbone model first and ensure float32
    backbone = AutoModel.from_pretrained(
        backbone_model_name, 
        trust_remote_code=True,
        cache_dir="models",
        torch_dtype=torch.float32
    )
    
    print("Creating HF-compatible config...")
    # Create HF config with all parameters
    hf_config = ESGHierarchicalConfig(
        backbone_model_name=backbone_model_name,
        num_esg_classes=config.num_esg_classes,
        num_e_sub_classes=config.num_e_sub_classes,
        num_s_sub_classes=config.num_s_sub_classes,
        num_g_sub_classes=config.num_g_sub_classes,
        shared_hidden_dim_factor=config.classification_config['architecture']['shared_hidden_dim_factor'],
        head_hidden_dim_factor=config.classification_config['architecture']['head_hidden_dim_factor'],
        dropout_rate=config.classification_config['architecture']['dropout_rate'],
        freeze_backbone=config.classification_config['architecture']['freeze_gte'],
        esg_categories=config.esg_categories,
        sub_factors_e=config.sub_factors_e,
        sub_factors_s=config.sub_factors_s,
        sub_factors_g=config.sub_factors_g,
        final_others_label=config.final_others_label,
        max_length=config.classification_config['max_length']
    )
    
    print("Initializing HF model with backbone...")
    # Initialize HF model with the backbone
    hf_model = ESGHierarchicalForSequenceClassification(hf_config, backbone=backbone)
    
    print(f"Loading trained weights from {weights_path}...")
    # Load the trained state dict
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    trained_state_dict = torch.load(weights_path, map_location='cpu')
    
    print("Converting weight names to HF format...")
    # The weights should map directly since we use the same architecture
    # Ensure all weights are in float32 to avoid dtype mismatch
    hf_state_dict = {}
    for key, value in trained_state_dict.items():
        # Convert all weights to float32 for consistency
        hf_state_dict[key] = value.float() if value.dtype == torch.float16 else value
    
    # Load the converted weights
    hf_model.load_state_dict(hf_state_dict, strict=True)
    
    # Ensure entire model is in float32
    hf_model = hf_model.float()
    print("Weights loaded successfully!")
    
    return hf_model, hf_config

def upload_to_hub(model, config, tokenizer, repo_name: str, private: bool = False):
    """
    Upload model to Hugging Face Hub
    """
    print(f"Preparing to upload to {repo_name}...")
    
    # Create temporary directory for saving
    temp_dir = "./temp_model_upload"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print("Saving model and config...")
        # Save model and config
        model.save_pretrained(temp_dir)
        
        # Save tokenizer (we'll use the original GTE tokenizer)
        tokenizer.save_pretrained(temp_dir)
        
        # Create model card
        model_card_content = create_model_card(repo_name)
        with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)
        
        print("Uploading to Hugging Face Hub...")
        # Upload to hub
        api = HfApi()
        api.create_repo(repo_name, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="model"
        )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/{repo_name}")
        
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    """Main function to run the upload process"""
    print("🚀 Starting ESG Model Upload Process...")
    
    # Configuration
    WEIGHTS_PATH = config.classification_config['weights_path']
    BACKBONE_MODEL_NAME = config.classification_config['base_model']
    
    # You need to set these
    REPO_NAME = input("Enter your Hugging Face repo name (format: username/model-name): ").strip()
    if not REPO_NAME or '/' not in REPO_NAME:
        print("❌ Invalid repo name. Format should be: username/model-name")
        return
    
    PRIVATE = input("Make repository private? (y/N): ").strip().lower() == 'y'
    
    try:
        # Load and convert model
        print("\n📦 Converting model...")
        hf_model, hf_config = load_and_convert_model(WEIGHTS_PATH, BACKBONE_MODEL_NAME)
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BACKBONE_MODEL_NAME,
            trust_remote_code=True
        )
        
        # Test the model
        print("🧪 Testing converted model...")
        test_text = "The company has implemented renewable energy solutions."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=4096)
        
        with torch.no_grad():
            esg_factor, sub_factor = hf_model.predict(inputs.input_ids, inputs.attention_mask)
            print(f"✅ Test prediction - ESG: {esg_factor}, Sub-factor: {sub_factor}")
        
        # Upload to hub
        print("\n🚀 Uploading to Hugging Face Hub...")
        upload_to_hub(hf_model, hf_config, tokenizer, REPO_NAME, PRIVATE)
        
        print("\n🎉 Upload completed successfully!")
        print(f"📁 Your model is available at: https://huggingface.co/{REPO_NAME}")
        print("\n💡 Usage example:")
        print(f"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("{REPO_NAME}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base")

text = "Your ESG text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
esg_factor, sub_factor = model.predict(inputs.input_ids, inputs.attention_mask)
        """)
        
    except Exception as e:
        print(f"❌ Error during upload process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
