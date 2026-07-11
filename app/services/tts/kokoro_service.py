import io

import numpy as np
import soundfile as sf
import torch
from fastapi import HTTPException, status
from kokoro import KPipeline

from app.schemas.TtsRequest import VoiceVariant
from app.services.tts.model_downloader import get_model_directory, load_kokoro_pipeline


class KokoroService:

    SAMPLE_RATE = 24000

    def __init__(self):

        try:
            # KPipeline otherwise creates its own Hub-backed KModel.
            model, device = load_kokoro_pipeline()

            self.pipelines = {
                "a": KPipeline(
                    lang_code="a",
                    model=model,
                    device=device,
                ),
                "b": KPipeline(
                    lang_code="b",
                    model=model,
                    device=device,
                ),
            }
            self._load_local_voices()

            print("Kokoro pipelines loaded successfully.")

        except Exception as ex:
            raise RuntimeError(
                f"Failed to initialize Kokoro pipelines: {ex}"
            ) from ex

    def _load_local_voices(self) -> None:
        """Preload supported voices so KPipeline never fetches them from the Hub."""
        voices_directory = get_model_directory() / "voices"
        for voice in VoiceVariant:
            voice_path = voices_directory / f"{voice.value}.pt"
            if not voice_path.is_file():
                raise FileNotFoundError(f"Kokoro voice file is missing: {voice_path}")
            self.pipelines[voice.value[0]].voices[voice.value] = torch.load(
                voice_path,
                map_location="cpu",
                weights_only=True,
            )

    def generate_audio(
        self,
        text: str,
        voice: VoiceVariant,
        speed: float = 1.0,
    ) -> io.BytesIO:

        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty.",
            )

        pipeline = self.pipelines.get(voice.value[0])

        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported voice: {voice.value}",
            )

        try:

            generator = pipeline(
                text=text,
                voice=voice.value,
                speed=speed,
                split_pattern=r"\n+",
            )

            audio_chunks = [
                audio
                for _, _, audio in generator
                if audio is not None and len(audio) > 0
            ]

            if not audio_chunks:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No audio generated.",
                )

            final_audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )

            output = io.BytesIO()

            sf.write(
                output,
                final_audio,
                samplerate=self.SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16",
            )

            output.seek(0)

            return output

        except HTTPException:
            raise

        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS generation failed: {ex}",
            ) from ex


kokoro_service = KokoroService()