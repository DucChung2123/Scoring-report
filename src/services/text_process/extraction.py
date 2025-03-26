import PyPDF2

class PDFTextExtractor:
    # def __init__(self, pdf_path: str):
    #     self.pdf_path = pdf_path
    
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

# if __name__ == "__main__":
#     pdf_path = "data/BCTN+BIDV+2021.pdf"
#     ouput_path = f"output/{pdf_path[5:-4]}.txt"
#     pdf_text = PDFTextExtractor.extract_text_from_pdf(pdf_path, ouput_path)
    # print(pdf_text)