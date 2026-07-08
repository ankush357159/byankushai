from pydantic import BaseModel, Field
class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Image generation prompt"
    )