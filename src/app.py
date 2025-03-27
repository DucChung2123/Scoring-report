from fastapi import FastAPI
from src.core.config import settings
from src.routers import ESG_search
from contextlib import asynccontextmanager
from src.services.search.embedding import Embedding

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    Embedding.initialize()
    print("Embedding initialized")
    yield
    
    print("bYe!")
    
app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan
)

app.include_router(ESG_search.router)

@app.get("/check-health")
def check_health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )