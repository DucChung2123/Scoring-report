from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

# Import the model manager and routers
from src.api.ai_models.model_manager import model_manager
from src.api.routers import health, scoring, classification
from src.api.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.
    Models are loaded when the app starts and before it begins accepting requests.
    """
    # Load all models on startup using the model manager
    await model_manager.load_all()
    yield
    
    # Cleanup on shutdown (if needed)
    print("Shutting down API, cleaning up resources...")

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="ESG Model API", 
    description="API endpoints for ESG scoring and classification", 
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(health.router)
app.include_router(scoring.router)
app.include_router(classification.router)

if __name__ == "__main__":
    print("Starting ESG Model API server...")
    # Get host and port from app config
    host = settings.APP_CONF.get('app', {}).get('host', '0.0.0.0')
    port = settings.APP_CONF.get('app', {}).get('port', 2003)
    uvicorn.run("src.api.app:app", host=host, port=port)
