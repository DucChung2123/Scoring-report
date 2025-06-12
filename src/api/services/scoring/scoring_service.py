from fastapi import HTTPException
from src.api.ai_models.model_manager import model_manager

class ScoringService:
    """Service for ESG scoring operations"""
    
    @staticmethod
    def score_text(text: str, factor: str) -> float:
        """Score a text for a given ESG factor"""
        try:
            return model_manager.score_text(text, factor)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Global service instance
scoring_service = ScoringService()
