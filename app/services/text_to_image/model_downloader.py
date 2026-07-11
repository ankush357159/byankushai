import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from diffusers import AutoPipelineForText2Image
from huggingface_hub import snapshot_download

load_dotenv()


MODEL_ID = "stabilityai/sd-turbo"


def load_image_pipeline() -> tuple[Any, str]:
    """
    Downloads model weights if necessary and returns a loaded pipeline.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "variant": "fp16",
        "use_safetensors": True,
    }

    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        kwargs["token"] = hf_token
        print("Hugging Face token successfully injected.")
    else:
        print("Warning: HF_TOKEN not found.")

    # Keep path behavior identical to the original unsplit service module.
    project_root = Path(__file__).resolve().parents[3]
    local_path = project_root / "model_weights" / "sd-turbo"

    download_kwargs = {
        "repo_id": MODEL_ID,
        "local_dir": str(local_path),
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

    if local_path.exists():
        print(f"Loading model from local directory: {local_path}")
        kwargs["local_files_only"] = True
    else:
        print("Downloading model weights...")

        snapshot_download(**download_kwargs)

        kwargs["local_files_only"] = True

    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            str(local_path),
            **kwargs,
        )

    except RuntimeError as exc:

        if "ignore_mismatched_sizes" not in str(exc):
            raise

        print("Retrying with ignore_mismatched_sizes=True")

        kwargs["ignore_mismatched_sizes"] = True

        pipeline = AutoPipelineForText2Image.from_pretrained(
            str(local_path),
            **kwargs,
        )

    except OSError as exc:

        if "config.json" not in str(exc):
            raise

        print("Incomplete model cache detected. Re-downloading...")

        snapshot_download(**download_kwargs)

        pipeline = AutoPipelineForText2Image.from_pretrained(
            str(local_path),
            **kwargs,
        )

    pipeline.to(device)

    if device == "cpu":
        pipeline.enable_attention_slicing()
        torch.set_num_threads(8)

    pipeline.set_progress_bar_config(disable=True)

    print(f"Model loaded on {device.upper()}")

    return pipeline, device