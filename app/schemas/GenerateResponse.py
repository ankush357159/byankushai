from pydantic import BaseModel
class GenerateResponse(BaseModel):
    prompt: str
    width: int
    height: int
    device: str
    generation_time_seconds: float
    image_base64: str