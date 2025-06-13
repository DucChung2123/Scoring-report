from fastapi import HTTPException
from typing import Tuple
from src.api.ai_models.model_manager import model_manager

class ClassificationService:
    """Service for ESG sub-factor classification operations"""
    
    @staticmethod
    def classify_text(text: str) -> Tuple[str, str, float, float]:
        """Classify a text to predict its ESG factor and sub-factor with probabilities"""
        try:
            return model_manager.classify_text(text)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Global service instance
classification_service = ClassificationService()
