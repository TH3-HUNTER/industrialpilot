"""
IndustrialPilot — Main Entry Point
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.db.database import init_db, clear_all_data
from backend.api.routes import router

app = FastAPI(title="IndustrialPilot", version="1.0.0", docs_url="/docs")
app.include_router(router)

FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND     = FRONTEND_DIR / "index.html"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_dashboard():
    return FileResponse(str(FRONTEND))


@app.on_event("startup")
async def on_startup():
    init_db()
    clear_all_data()   # ← fresh start every restart
    print("\n✅  IndustrialPilot is running — data cleared")
    print("🌐  Dashboard → http://localhost:8000")
    print("📖  API Docs  → http://localhost:8000/docs\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=True
    )
