from fastapi import FastAPI
from src.core.config import settings
app = FastAPI(
    title=settings.APP_NAME,
)

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