import PyPDF2
import re
import os
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
    def extract_text_from_pdf(pdf_path: str,
                              output_path: str = None) -> str:
        with open(pdf_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            text = ""
            pages = pdf.pages
            for page_num in range(len(pages)):
                page = pages[page_num]
                text += page.extract_text()
            if output_path:
                with open(output_path, "w") as out_file:
                    out_file.write(text)
        return text
        
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
#     ouput_path = f"output/{pdf_path[5:-4]}.txt"
#     pdf_text = PDFTextExtractor.extract_text_from_pdf(pdf_path, ouput_path)
#     print(pdf_text)
    
#     sentences = PDFTextExtractor.sentence_segmentation(pdf_text)
#     with open(f"output/sentence_{pdf_path[5:-4]}.jsonl", "w") as out_file:
#         for sentence in sentences:
#             s = sentence.replace("\n", " ") 
#             out_file.write(f"\"sentence\":\"{s}\"\n")