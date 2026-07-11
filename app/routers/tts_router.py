from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.TtsRequest import TtsRequest
from app.services.tts.kokoro_service import kokoro_service

router = APIRouter(
    prefix="/tts",
    tags=["Text To Speech"],
)


@router.post("/generate")
async def generate(request: TtsRequest):

    audio = kokoro_service.generate_audio(
        text=request.text,
        voice=request.voice,
        speed=request.speed,
    )

    return StreamingResponse(
        audio,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav"
        },
    )