import os

from dotenv import load_dotenv
from fastapi import HTTPException, status
from google import genai
from google.genai import errors

from app.schemas.PromptRequest import PromptRequest
from app.schemas.PromptResponse import PromptResponse

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

ai_client = genai.Client(api_key=api_key)


async def generate_response(payload: PromptRequest) -> PromptResponse:
    """
    Sends a prompt to Gemini and returns the generated response.
    """

    model_name = payload.model_name or "gemini-3.5-flash"

    try:
        response = ai_client.models.generate_content(
            model=model_name,
            contents=payload.prompt,
        )

        if not response.text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The model returned an empty response.",
            )

        return PromptResponse(
            generated_text=response.text,
            model_used=model_name,
        )

    except errors.APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error: {str(exc)}",
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        # Log exc here if a logging framework is configured.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from exc