from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    """Schema for incoming user generation requests."""
    prompt: str = Field(
        ...,
        description="The text prompt to send to the Gemini model.",
        examples=["Explain quantum computing in simple terms."]
    )
    model_name: str = Field(
        default="gemini-3.5-flash",
        description="The specific Gemini model to target.",
        examples=["gemini-3.5-flash", "gemini-2.5-pro"]
    )