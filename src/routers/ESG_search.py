from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional, Dict, List, Any
from src.services.text_process.extraction import PDFTextExtractor
from src.services.search.engine import SearchEngine
from src.services.search.ESG_search import ESGSearch

router = APIRouter(prefix="/api/v1/esg", tags=["ESG"])

@router.post("/esg-extract")
async def esg_extract(
    upload_file: UploadFile = File(...),
    top_k: Optional[int] = Query(5, description="Số lượng kết quả trả về cho mỗi yếu tố ESG", ge=1, le=50),
    threshold: Optional[float] = Query(0.5, description="Ngưỡng điểm tương đồng tối thiểu (0-1)", ge=0.0, le=1.0)
):
    """
    Trích xuất văn bản từ tệp PDF tải lên, phân tích nội dung liên quan đến ESG 
    (Environmental, Social, and Governance) và trả về kết quả phân tích.
    
    Parameters:
    - **upload_file**: Tệp PDF cần phân tích (bắt buộc)
    - **top_k**: Số lượng kết quả trả về cho mỗi yếu tố ESG (mặc định: 5)
    - **threshold**: Ngưỡng điểm tương đồng tối thiểu để lấy kết quả (0-1, mặc định: 0.5)
    
    Returns:
    - Kết quả phân tích ESG của tài liệu với các thông tin phân loại theo Environmental, Social, và Governance
    """
    try:
        # Lưu và xử lý file PDF
        file_name, file_path = PDFTextExtractor.save_to_upload(upload_file)
        
        # Trích xuất và làm sạch văn bản
        texts = PDFTextExtractor.extract_text_from_pdf(file_path)
        clean_texts = PDFTextExtractor.clean_vietnamese_text(texts)
        
        # Khởi tạo SearchEngine với văn bản đã làm sạch
        search_engine = SearchEngine(clean_texts)
        
        # Tìm kiếm ESG với các tham số đã cung cấp
        results = ESGSearch.esg_search(
            engine=search_engine,
            top_k=top_k,
            threshold=threshold
        )
        
        # Chuẩn bị kết quả trả về
        response = {
            "file_name": file_name,
            "total_paragraphs": len(clean_texts),
            "esg_results": results,
            "search_parameters": {
                "top_k": top_k,
                "threshold": threshold
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")
