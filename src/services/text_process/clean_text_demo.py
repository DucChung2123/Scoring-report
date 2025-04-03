import os
import json
import sys
from pathlib import Path

# Thêm đường dẫn gốc của dự án vào sys.path để import được các module
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.services.text_process.extraction import PDFTextExtractor

def clean_and_save_text(input_json_path: str, output_file_path: str = None):
    """
    Đọc file JSON từ đầu ra của extract_text_from_pdf, 
    làm sạch dữ liệu và lưu vào file mới.
    
    Args:
        input_json_path (str): Đường dẫn đến file JSON đầu vào
        output_file_path (str, optional): Đường dẫn đến file đầu ra. Nếu None, sẽ tạo tên file tự động
    """
    # Đọc dữ liệu từ file JSON
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
        return
    
    # Làm sạch dữ liệu
    cleaned_texts = PDFTextExtractor.clean_vietnamese_text(json_data)
    
    # Tạo tên file đầu ra nếu không được cung cấp
    if not output_file_path:
        base_name = os.path.basename(input_json_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_dir = os.path.dirname(input_json_path)
        output_file_path = os.path.join(output_dir, f"cleaned_{name_without_ext}.json")
    
    # Lưu dữ liệu đã làm sạch vào file mới
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_texts, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu dữ liệu đã làm sạch vào {output_file_path}")
        print(f"Số đoạn văn bản sau khi làm sạch: {len(cleaned_texts)}")
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")

if __name__ == "__main__":
    # Đường dẫn file JSON đầu vào
    input_json = "output/GVR_Baocaothuongnien_2023.json"
    
    # Làm sạch và lưu dữ liệu
    clean_and_save_text(input_json)
