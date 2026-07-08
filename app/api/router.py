from fastapi import APIRouter

from app.routers.image_router import router as image_router

api_router = APIRouter(prefix="/api")
api_router.include_router(image_router)

