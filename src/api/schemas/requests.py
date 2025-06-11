from pydantic import BaseModel, Field
from typing import Literal

class ScoreRequest(BaseModel):
    text: str = Field(..., description="The text to analyze")
    factor: Literal["E", "S", "G"] = Field(..., description="The factor to score (E, S, or G)")

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="The text to classify")
