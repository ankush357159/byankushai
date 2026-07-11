import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from kokoro import KModel

from app.schemas.VoiceVariant import VoiceVariant

load_dotenv()

MODEL_ID = "hexgrad/Kokoro-82M"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DOWNLOAD_ATTEMPTS = 3


def get_model_directory() -> Path:
    """Return the project-local directory used for Kokoro model assets."""
    return Path(__file__).resolve().parents[3] / "model_weights" / "kokoro_82M"


def load_kokoro_pipeline() -> tuple[KModel, str]:
    """Load one Kokoro model from local files, downloading missing assets once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_path = get_model_directory()
    voice_patterns = [f"voices/{variant.value}.pt" for variant in VoiceVariant]
    required_files = [
        local_path / "config.json",
        local_path / "kokoro-v1_0.pth",
        *(local_path / voice_path for voice_path in voice_patterns),
    ]
    missing_files = [
        path.relative_to(local_path).as_posix()
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        print(f"Downloading missing Kokoro assets: {', '.join(missing_files)}")
        # Do not inherit HF_ENDPOINT from .env. This project currently uses a mirror
        # for other models, but that mirror can be unavailable or incomplete for Kokoro.
        endpoint = os.getenv("KOKORO_HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
        download_kwargs = {
            "repo_id": MODEL_ID,
            "local_dir": str(local_path),
            "revision": "main",
            "endpoint": endpoint,
            "allow_patterns": ["config.json", "kokoro-v1_0.pth", *voice_patterns],
        }
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            download_kwargs["token"] = hf_token
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                snapshot_download(**download_kwargs)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < DOWNLOAD_ATTEMPTS:
                    print(
                        f"Kokoro download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed; "
                        "retrying..."
                    )
                    time.sleep(attempt)

        if last_error is not None:
            raise RuntimeError(
                "Kokoro assets are incomplete and could not be downloaded. "
                f"Missing: {', '.join(missing_files)}. Check Internet access or download "
                f"{MODEL_ID} into {local_path}. Endpoint: {endpoint}. "
                f"Hub error: {last_error}"
            ) from last_error
    else:
        print(f"Loading Kokoro assets from local directory: {local_path}")

    missing_files = [
        path.relative_to(local_path).as_posix()
        for path in required_files
        if not path.is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "Kokoro download completed without all required assets. "
            f"Missing: {', '.join(missing_files)}."
        )

    if device == "cpu":
        torch.set_num_threads(8)

    model = KModel(
        config=str(local_path / "config.json"),
        model=str(local_path / "kokoro-v1_0.pth"),
    ).to(device).eval()
    print(f"Kokoro model loaded on {device.upper()}")
    return model, device
