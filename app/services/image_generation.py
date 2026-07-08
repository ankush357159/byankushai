import base64
import io
import time

import torch
from diffusers import AutoPipelineForText2Image

from app.schemas.GenerateRequest import GenerateRequest
from app.schemas.GenerateResponse import GenerateResponse



class ImageGenerationService:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        kwargs = {
            "torch_dtype": self.dtype
        }

        if self.device == "cuda":
            kwargs["variant"] = "fp16"

        print("Loading model...")

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            **kwargs
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