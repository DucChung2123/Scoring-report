from fastapi import APIRouter
from typing import List
from src.api.schemas.requests import ScoreRequest
from src.api.schemas.responses import ScoreResponse
from src.api.services.scoring.scoring_service import scoring_service

router = APIRouter(prefix="/score", tags=["scoring"])

@router.post("", response_model=ScoreResponse)
def score_single(request: ScoreRequest):
    """Score a single text for a specific ESG factor."""
    score = scoring_service.score_text(request.text, request.factor)
    return {"score": score}

@router.post("_batch", response_model=List[ScoreResponse])
def score_multiple(requests: List[ScoreRequest]):
    """Score multiple texts for specific ESG factors."""
    results = []
    for request in requests:
        score = scoring_service.score_text(request.text, request.factor)
        results.append({"score": score})
    return results
