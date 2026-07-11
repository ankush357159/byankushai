from fastapi import APIRouter, HTTPException

from app.schemas.PromptRequest import PromptRequest
from app.schemas.PromptResponse import PromptResponse
from app.services.genai.generate_response import generate_response

MODEL_ID = "gemini-3.5-flash"

router = APIRouter(
    prefix="/text",
    tags=["Text Generation"]
)


@router.get("/health")
async def health():

    return {
        "status": "UP",
        "provider": "Google Gemini",
        "default_model": MODEL_ID,
    }


@router.post(
    "/generate",
    response_model=PromptResponse,
)
async def generate_text(
    request: PromptRequest,
):

    try:
        return await generate_response(request)

    except HTTPException:
        # Preserve HTTPExceptions raised by the service.
        raise

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex),
        )