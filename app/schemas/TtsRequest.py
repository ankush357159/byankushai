
from pydantic import BaseModel, Field

from app.schemas.VoiceVariant import VoiceVariant


class TtsRequest(BaseModel):
    """Request payload for Kokoro TTS."""

    text: str = Field(
        ...,
        min_length=1,
        description="Text to convert into speech.",
        examples=["Hello. Welcome to Kokoro text to speech."],
    )

    voice: VoiceVariant = Field(
        default=VoiceVariant.AMERICAN_FEMALE_WARM,
        description="Voice variant.",
    )

    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed.",
    )