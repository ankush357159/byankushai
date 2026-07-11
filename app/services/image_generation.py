import base64
import io
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

import torch
from diffusers import AutoPipelineForText2Image

from app.schemas.GenerateResponse import GenerateResponse

load_dotenv()

class ImageGenerationService:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dtype = (torch.float16 if self.device == "cuda" else torch.float32)

        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "variant": "fp16",
            "use_safetensors": True,
        }

        # Fetch the token from your environment variables
        hf_token = os.getenv("HF_TOKEN")

        # Inject the token into the keyword arguments if it exists
        if hf_token:
            kwargs["token"] = hf_token
            print("Hugging Face token successfully injected.")
        else:
            print("Warning: HF_TOKEN not found in .env file.")

        model_id = "stabilityai/sd-turbo"

        # Build a stable absolute path so startup works regardless of cwd.
        project_root = Path(__file__).resolve().parents[2]
        local_path = str(project_root / "model_weights" / "sd-turbo")

        download_kwargs = {
            "repo_id": model_id,
            "local_dir": local_path,
            "revision": "main",
            "token": hf_token,
            "allow_patterns": [
                "*.json",
                "*.txt",
                "*fp16.safetensors",
            ],
            "ignore_patterns": [
                "sd_turbo.safetensors",
                "*.bin",
                "*.ckpt",
            ],
        }

        if os.path.exists(local_path) and os.path.isdir(local_path):
            print(f"Loading model from local project workspace: {local_path}")
            model_target = local_path
            kwargs["local_files_only"] = True
        else:
            print("Local weights not found. Downloading sd-turbo...")

            snapshot_download(**download_kwargs)

            model_target = local_path
            kwargs["local_files_only"] = True


        print(f"Loading model from: {model_target}")

        try:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                model_target,
                **kwargs,
            )
        except RuntimeError as exc:
            # Some torch/transformers combos report classifier size mismatch at load time.
            if "ignore_mismatched_sizes" not in str(exc):
                raise
            print("Retrying model load with ignore_mismatched_sizes=True")
            kwargs["ignore_mismatched_sizes"] = True
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                model_target,
                **kwargs,
            )
        except OSError as exc:
            # Recover from partial/corrupted HF cache by downloading a clean local copy.
            if "config.json" not in str(exc):
                raise
            print("Detected incomplete model cache. Re-downloading model weights...")

            snapshot_download(**download_kwargs)

            model_target = local_path
            kwargs["local_files_only"] = True
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                model_target,
                **kwargs,
            )

        self.pipeline.to(self.device)

        if self.device == "cpu":
            self.pipeline.enable_attention_slicing()
            torch.set_num_threads(8)

        self.pipeline.set_progress_bar_config(disable=True)

        print(f"Model loaded on {self.device.upper()}")

    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512
    ) -> GenerateResponse:

        start = time.perf_counter()

        with torch.inference_mode():

            image = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=1,
                guidance_scale=0.0
            ).images[0]

        buffer = io.BytesIO()

        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        elapsed = round(
            time.perf_counter() - start,
            2
        )

        return GenerateResponse(
            prompt=prompt,
            width=width,
            height=height,
            device=self.device,
            generation_time_seconds=elapsed,
            image_base64=encoded
        )


image_generation_service = ImageGenerationService()