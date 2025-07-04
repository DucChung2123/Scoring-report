from fastapi import APIRouter
from typing import List
from src.api.schemas.requests import ClassifyRequest
from src.api.schemas.responses import ClassifyResponse
from src.api.services.classification.classification_service import classification_service

router = APIRouter(prefix="/classify_sub_factor", tags=["classification"])

@router.post("", response_model=ClassifyResponse)
def classify_single(request: ClassifyRequest):
    """Classify a single text to predict its ESG factor and sub-factor with probabilities."""
    factor, sub_factor, factor_prob, sub_factor_prob = classification_service.classify_text(request.text)
    return {
        "factor": factor, 
        "sub_factor": sub_factor,
        "factor_probability": factor_prob,
        "sub_factor_probability": sub_factor_prob
    }

@router.post("_batch", response_model=List[ClassifyResponse])
def classify_multiple(requests: List[ClassifyRequest]):
    """Classify multiple texts to predict their ESG factors and sub-factors with probabilities."""
    results = []
    for request in requests:
        factor, sub_factor, factor_prob, sub_factor_prob = classification_service.classify_text(request.text)
        results.append({
            "factor": factor, 
            "sub_factor": sub_factor,
            "factor_probability": factor_prob,
            "sub_factor_probability": sub_factor_prob
        })
    return results
