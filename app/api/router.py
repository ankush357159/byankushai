from fastapi import APIRouter

from app.routers import gemini_router, tts_router
from app.routers.image_router import router as image_router

api_router = APIRouter(prefix="/api")
api_router.include_router(image_router)
api_router.include_router(gemini_router.router)

api_router.include_router(tts_router.router)