# Hello, I'm Chung, AI Engineer in MISA JSC!
import torch

# --- Label definitions and mappings ---
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

# --- Constants ---
IGNORE_INDEX = -100
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- File paths ---
DATA_PATH = "/mnt/data/work/llm/hdchung/new_datn/datn/data/processed/80k_sample_extracted_text_processed.jsonl"
BEST_MODEL_SAVE_PATH = "output/20-5-full-data_best_hierarchical_mtl_4096_esg_model_state.pth"

# --- Training parameters ---
BATCH_SIZE = 8
PHASE_1_EPOCHS = 3
PHASE_2_EPOCHS = 12
TOTAL_EPOCHS = PHASE_1_EPOCHS + PHASE_2_EPOCHS

# --- Loss weights ---
LOSS_WEIGHTS = {"esg": 1.0, "e_sub": 0.7, "s_sub": 0.7, "g_sub": 0.7}
