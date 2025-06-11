from pydantic import BaseModel, Field

class ScoreResponse(BaseModel):
    score: float = Field(..., description="The probability score for the specified factor")

class ClassifyResponse(BaseModel):
    factor: str = Field(..., description="The predicted ESG factor (E, S, G, or Others_ESG)")
    sub_factor: str = Field(..., description="The predicted sub-factor")
