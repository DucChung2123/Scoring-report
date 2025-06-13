from .scoring_model import ScoringModelManager
from .classification_model import ClassificationModelManager

class ModelManager:
    """
    Main coordinator for all AI models.
    Orchestrates loading and provides unified interface.
    """
    
    def __init__(self):
        self.scoring = ScoringModelManager()
        self.classification = ClassificationModelManager()
    
    async def load_all(self) -> None:
        """Load all models"""
        await self.scoring.load()
        await self.classification.load()
        print("All models loaded successfully. API ready to handle requests.")
    
    def score_text(self, text: str, factor: str) -> float:
        """Score text for ESG factor"""
        return self.scoring.predict(text, factor)
    
    def classify_text(self, text: str) -> tuple[str, str, float, float]:
        """Classify text for ESG factor and sub-factor with probabilities"""
        return self.classification.predict(text)

# Global model manager instance
model_manager = ModelManager()
