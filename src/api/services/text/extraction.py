import re
import os
import json
from pdfminer.high_level import extract_text, extract_pages

class PDFTextExtractor:
    # def __init__(self, pdf_path: str):
    #     self.pdf_path = pdf_path
    
    @staticmethod
    def save_to_upload(file):
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        return file.filename, file_path
        
    @staticmethod
    def clean_vietnamese_text(json_objects: list) -> list:
        """
        Làm sạch văn bản tiếng Việt từ danh sách JSON objects, bao gồm:
        - Ghép văn bản từ nhiều trang
        - Loại bỏ số trang, header, và các phần thừa khác
        - Xử lý ký tự ngắt dòng
        - Gộp các câu ngắn thành đoạn có nghĩa
        
        Args:
            json_objects (list): Danh sách các JSON object, mỗi object chứa thông tin của một trang
                với cấu trúc {"page": số_trang, "text": nội_dung_văn_bản}
                
        Returns:
            list: Danh sách các đoạn văn bản đã được làm sạch
        """
        # Kiểm tra nếu không có dữ liệu
        if not json_objects:
            return []
        
        # Thu thập tất cả văn bản thô từ các trang
        all_text = ""
        for obj in json_objects:
            text = obj.get('text', '').strip()
            if text:
                all_text += text + "\n\n"
        
        # Tiền xử lý văn bản
        # 1. Loại bỏ số trang độc lập (các số có 1-3 chữ số ở dòng riêng)
        all_text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n\n', all_text)
        
        # 2. Loại bỏ các header lặp lại (viết hoa hoàn toàn và xuất hiện nhiều lần)
        header_pattern = r'([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ\s]{10,})'
        headers = re.findall(header_pattern, all_text)
        for header in set(headers):
            # Nếu header xuất hiện nhiều lần, chỉ giữ lần xuất hiện đầu tiên
            if all_text.count(header) > 1:
                # Tìm vị trí xuất hiện đầu tiên
                first_pos = all_text.find(header)
                # Thay thế các lần xuất hiện tiếp theo bằng khoảng trống
                all_text = all_text[:first_pos+len(header)] + all_text[first_pos+len(header):].replace(header, '')
        
        # 3. Chuẩn hóa ngắt dòng và khoảng trống
        all_text = re.sub(r'\n{3,}', '\n\n', all_text)  # Thay thế 3+ xuống dòng bằng 2 xuống dòng
        all_text = re.sub(r'\s+', ' ', all_text)  # Thay thế nhiều khoảng trắng bằng 1 khoảng trắng
        
        # Chia văn bản thành các đoạn dựa trên các khoảng trống lớn
        paragraphs_raw = re.split(r'\n\s*\n', all_text)
        
        # Xử lý và làm sạch từng đoạn
        paragraphs = []
        current_paragraph = ""
        
        for p in paragraphs_raw:
            p = p.strip()
            if not p:
                continue
            
            # Nếu đoạn quá ngắn và không kết thúc bằng dấu câu
            if len(p) < 100 and not re.search(r'[.!?]$', p):
                if current_paragraph:
                    current_paragraph += " " + p
                else:
                    current_paragraph = p
            else:
                # Nếu đoạn đủ dài hoặc kết thúc bằng dấu câu
                if current_paragraph:
                    current_paragraph += " " + p
                    paragraphs.append(current_paragraph)
                    current_paragraph = ""
                else:
                    paragraphs.append(p)
        
        # Xử lý đoạn cuối cùng nếu còn
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # Kiểm tra và xử lý các đoạn quá dài
        max_paragraph_length = 1000  # Độ dài tối đa của một đoạn (khoảng 200 từ)
        processed_paragraphs = []
        
        for p in paragraphs:
            if len(p) > max_paragraph_length:
                # Tìm các vị trí có thể chia đoạn (kết thúc bằng dấu chấm)
                sentence_ends = [m.end() for m in re.finditer(r'\.(?=\s)', p)]
                
                if sentence_ends:
                    # Chia đoạn dài thành nhiều đoạn nhỏ hơn
                    start = 0
                    for i, end in enumerate(sentence_ends):
                        if end - start >= max_paragraph_length / 2 or i == len(sentence_ends) - 1:
                            sub_p = p[start:end].strip()
                            if len(sub_p) > 50:  # Chỉ lấy các đoạn có ý nghĩa
                                processed_paragraphs.append(sub_p)
                            start = end
                    
                    # Xử lý phần còn lại nếu có
                    if start < len(p):
                        remaining = p[start:].strip()
                        if len(remaining) > 50:
                            processed_paragraphs.append(remaining)
                else:
                    # Nếu không thể tìm thấy vị trí tốt để chia, giữ nguyên đoạn
                    processed_paragraphs.append(p)
            else:
                processed_paragraphs.append(p)
        
        # Làm sạch lần cuối các đoạn văn bản
        final_paragraphs = []
        
        for p in processed_paragraphs:
            # Loại bỏ các ký tự đặc biệt không cần thiết
            p = re.sub(r'[^\w\s\.,;:?!()"""''\-–—\[\]ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]', '', p)
            
            # Chuẩn hóa các dấu câu
            p = re.sub(r'\s+([.,;:?!)])', r'\1', p)
            p = re.sub(r'([(\["])\s+', r'\1', p)
            p = re.sub(r'\s+', ' ', p).strip()
            
            # Loại bỏ các đoạn chỉ có số liệu
            if re.match(r'^[\d\s.,:%/]+$', p):
                continue
                
            # Chỉ lấy các đoạn có ý nghĩa và đủ dài (> 50 ký tự)
            if len(p) > 50:
                # Đếm số lượng chữ cái tiếng Việt
                alpha_count = sum(1 for c in p if c.isalpha() or c in 'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ')
                # Chỉ lấy đoạn có ít nhất 50% là chữ cái
                if alpha_count / len(p) >= 0.5:
                    final_paragraphs.append(p)
        
        return final_paragraphs
    
    @staticmethod    
    def extract_text_from_pdf(pdf_path: str,
                              output_path: str = None) -> str:
        # with open(pdf_path, "rb") as file:
        #     pdf = PyPDF2.PdfReader(file)
        #     text = ""
        #     pages = pdf.pages
        #     for page_num in range(len(pages)):
        #         page = pages[page_num]
        #         text += page.extract_text()
        #     if output_path:
        #         with open(output_path, "w") as out_file:
        #             out_file.write(text)
        
        # Get page layouts
        page_layouts = list(extract_pages(pdf_path))
        results = []
        text = ""
        
        # Process each page layout
        for i, layout in enumerate(page_layouts):
            # Extract text from this page
            page_text = ""
            for element in layout:
                if hasattr(element, "get_text"):
                    page_text += element.get_text()
            
            # Add to results
            obj = {
                "page": i,
                "text": page_text
            }
            
            # Add to total text
            text += page_text
            results.append(obj)
        
        # Write to JSON file
        if output_path:
            with open(output_path, "w") as out_file:
                json.dump(results, out_file, indent=4, ensure_ascii=False)
                
        return results
        
    @staticmethod
    def sentence_segmentation(text: str) -> list[str]:
        """
        Extract sentences from text, clean them by:
        - Removing extra whitespace
        - Replacing newlines and other special characters
        - Filtering out sentences shorter than 20 characters
        - Filtering out sentences with less than 50% alphabetic characters
        
        Args:
            text (str): The input text to be processed
            
        Returns:
            list: A list of cleaned sentences with length >= 20 characters
                  and at least 50% alphabetic characters
        """
        # Define sentence boundaries (., !, ?) but exclude decimal numbers like 3.14
        # Negative lookbehind to avoid splitting on decimal points in numbers
        # This pattern handles periods not preceded by a digit or followed by a digit
        # as well as exclamation and question marks followed by space or end of string
        sentence_delimiters = r'(?<!\d)(?<=[.!?])(?!\d)\s+'
        
        # Split text into raw sentences
        raw_sentences = re.split(sentence_delimiters, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        
        for sentence in raw_sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Clean the sentence:
            # 1. Replace newlines with spaces
            cleaned = re.sub(r'\n+', ' ', sentence)
            
            # 2. Replace multiple spaces, tabs with single space
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # 3. Remove leading/trailing whitespace
            cleaned = cleaned.strip()
            
            # Skip sentences shorter than 20 characters
            if len(cleaned) < 20:
                continue
                
            # Count alphabetic characters in the sentence
            alpha_count = sum(1 for char in cleaned if char.isalpha())
            
            # Calculate the percentage of alphabetic characters
            alpha_percentage = alpha_count / len(cleaned) if len(cleaned) > 0 else 0
            
            # Filter out sentences with less than 50% alphabetic characters
            if alpha_percentage >= 0.5:
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
# if __name__ == "__main__":
#     pdf_path = "data/GVR_Baocaothuongnien_2023.pdf"
#     ouput_path = f"output/{pdf_path[5:-4]}.json"
#     pdf_text = PDFTextExtractor.extract_text_from_pdf(pdf_path, ouput_path)
#     # print(pdf_text)
    
#     sentences = PDFTextExtractor.sentence_segmentation(pdf_text)
#     with open(f"output/sentence_{pdf_path[5:-4]}.jsonl", "w") as out_file:
#         for sentence in sentences:
#             s = sentence.replace("\n", " ") 
#             out_file.write(f"\"sentence\":\"{s}\"\n")