from fastapi import APIRouter, UploadFile, File, HTTPException
from src.services.text_process.extraction import PDFTextExtractor
from src.services.search.engine import SearchEngine
from src.services.search.ESG_search import ESGSearch


    
router = APIRouter(prefix="/api/v1/esg", tags=["ESG"])

@router.post("/esg-extract")
async def esg_extract(upload_file: UploadFile = File(...)):
    """
    Extract text from an uploaded PDF file
    """
    # save
    file_name, file_path = PDFTextExtractor.save_to_upload(upload_file)
    texts = PDFTextExtractor.extract_text_from_pdf(file_path)
    sentences = PDFTextExtractor.sentence_segmentation(texts)
    print("Extracted sentences from PDF")
    SearchEngine_instance = SearchEngine(sentences)
    print("Search engine initialized")
    results = ESGSearch.esg_search(engine=SearchEngine_instance,
                                   top_k=10,
                                   threshold=0.5)
    
    return results
    
    
    