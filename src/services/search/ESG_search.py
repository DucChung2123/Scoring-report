from src.services.search.engine import SearchEngine
from src.core.config import settings

class ESGSearch:
    esg = settings.ESG
    
    @classmethod
    def esg_search(cls,
                   engine: SearchEngine,
                   top_k: int = 5,
                   threshold: float = 0.5):
        results = {}
        for category, factors in cls.esg.items():
            # Khởi tạo dictionary trống cho mỗi category
            if category not in results:
                results[category] = {}
                
            for factor, explanation in factors.items():
                print(f"search for: {category} - {factor}: {explanation}")
                results[category][factor] = engine.search(explanation, k=top_k, threshold=threshold)
        return results

# if __name__ == "__main__":
#     from src.services.text_process.extraction import PDFTextExtractor
#     pdf_path = "data/GVR_Baocaothuongnien_2023.pdf"
#     pdf_text = PDFTextExtractor.extract_text_from_pdf(pdf_path)
#     # print(pdf_text)

#     sentences = PDFTextExtractor.sentence_segmentation(pdf_text)
    
#     from src.services.search.engine import SearchEngine
#     SearchEngine_instance = SearchEngine(sentences)
#     results = ESGSearch.esg_search(engine=SearchEngine_instance,
#                                    top_k=20,
#                                    threshold=0.5)
#     print(results)
#     # SAVE TO JSON
#     import json
#     with open("data/esg_search_results.json", "w") as f:
#         json.dump(results, f, ensure_ascii=False, indent=4)