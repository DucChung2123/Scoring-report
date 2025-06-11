from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_root():
    """Root endpoint to verify the API is working."""
    return {"status": "alive", "message": "ESG Model API is running!"}
