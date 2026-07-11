from fastapi import APIRouter, HTTPException

from app.schemas.GenerateRequest import GenerateRequest
from app.schemas.GenerateResponse import GenerateResponse
from app.services.text_to_image.image_generation_service import image_generation_service
from app.services.text_to_image.model_downloader import MODEL_ID

router = APIRouter(
    prefix="/images",
    tags=["Images"]
)


@router.get("/health")
async def health():

    return {
        "status": "UP",
        "device": image_generation_service.device,
        "model": MODEL_ID
    }


@router.post(
    "/generate",
    response_model=GenerateResponse
)
async def generate_image(
    request: GenerateRequest
):

    try:
        return image_generation_service.generate(
            request.prompt
        )

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )