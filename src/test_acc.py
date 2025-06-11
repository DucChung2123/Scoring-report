import pandas as pd
import requests
from tqdm import tqdm
# real time accuracy check for ESG sub factor classification
path = "data/eval/ESG_data.jsonl"
url = 'http://localhost:8888/classify_sub_factor'

df = pd.read_json(path, lines=True)

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

count_subF = 0
total_processed = 0
count_F = 0
pbar = tqdm(df.iterrows(), total=len(df), desc="SubF - Acc: 0.00% | F - Acc: 0.00%")

for _, row in pbar:
    text = row["text"]
    json_data = {
        "text": text,
    }
    response = requests.post(url, headers=headers, json=json_data)
    
    if response.status_code == 200:
        result = response.json()
        if result["sub_factor"] == row["sub_factor"]:
            count_subF += 1
        if result["factor"] == row["factor"]:
            count_F += 1
    
    total_processed += 1
    subF_current_accuracy = (count_subF / total_processed) * 100
    F_current_accuracy = (count_F / total_processed) * 100
    # Update progress bar description with real-time accuracy
    pbar.set_description(f"SubF - Acc: {subF_current_accuracy:.2f}% | F - Acc: {F_current_accuracy:.2f}%")

pbar.close()
print(f"Final Sub-Factor Accuracy: {count_subF / total_processed * 100:.2f}%")
print(f"Final Factor Accuracy: {count_F / total_processed * 100:.2f}%")
