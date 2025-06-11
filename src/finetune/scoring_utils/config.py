# Hello, I'm Chung, AI Engineer in MISA JSC!
import torch

# --- File paths ---
DATA_PATH = "data/processed/ESG_data.jsonl"
OUTPUT_DIR = "./results_gte_classifier_dynamic"

# --- Model Configuration ---
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MODEL_MAX_LENGTH = 8192
CACHE_DIR = "models"

# --- Label Mapping ---
LABEL2ID = {"E": 0, "S": 1, "G": 2}
ID2LABEL = {0: "E", 1: "S", 2: "G"}
NUM_LABELS = len(LABEL2ID)

# --- Training Parameters ---
TRAIN_TEST_SPLIT_RATIO = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 3

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
