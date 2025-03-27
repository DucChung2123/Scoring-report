from pydantic import BaseModel, Field

class ESGSearchRequest(BaseModel):
    top_k: int = Field(5, description="The number of results to return for each factor")
    
class ESGSearchResponse(BaseModel):
    factor: str = Field(..., description="The ESG factor")
    results: list[str] = Field(..., description="The top k results for the factor")