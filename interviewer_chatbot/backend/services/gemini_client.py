import time
import json
import re
from typing import Type
from pydantic import BaseModel, Field, ValidationError
from models.gemini_model import GeminiModel
from utils.logger import setup_logger

logger = setup_logger(__name__)


class QuestionFeedback(BaseModel):
    """
    Pydantic model representing evaluation feedback for a question.

    Attributes:
        rating (int): Rating of the question (0-10).
        feedback (str): Descriptive feedback for the question.
    """

    rating: int = Field(0, ge=0, le=10)
    feedback: str = "No feedback"


class AnswerFeedback(BaseModel):
    """
    Pydantic model representing evaluation feedback for an answer.

    Attributes:
        rating (int): Rating of the answer (0-10).
        feedback (str): Descriptive feedback for the answer.
    """

    rating: int = Field(0, ge=0, le=10)
    feedback: str = "No feedback"


class GeminiClient:
    """
    Wrapper class for interacting with the Gemini LLM API.
    Provides retry mechanism and JSON validation using Pydantic models.
    """

    def __init__(self):
        """
        Initializes the GeminiClient with a GeminiModel instance.
        """
        self.model = GeminiModel

    def generate_content(self, prompt: str, retries: int = 3, delay: int = 5) -> str:
        """
        Generates text content from Gemini LLM for a given prompt.

        Args:
            prompt (str): The input prompt to send to Gemini LLM.
            retries (int, optional): Number of retry attempts if API fails. Default is 3.
            delay (int, optional): Delay in seconds between retries. Default is 5.

        Returns:
            str: The generated text from the model, or empty string on failure.
        """
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                logger.info(
                    f"Generated content for prompt (length {len(prompt)} chars)"
                )
                return response.text.strip() if response.text else ""
            except Exception as e:
                logger.exception(f"Gemini API error (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    logger.error("Gemini API failed after maximum retries")
                    return ""

    def safe_parse_json(
        self, response_text: str, model: Type[BaseModel] = QuestionFeedback
    ) -> dict:
        """
        Safely parses a JSON string response from Gemini LLM and validates it using a Pydantic model.
        Returns a default model dictionary if parsing or validation fails.

        Args:
            response_text (str): The raw JSON string returned by Gemini LLM.
            model (Type[BaseModel], optional): Pydantic model class to validate JSON. Default is QuestionFeedback.

        Returns:
            dict: Validated dictionary according to the Pydantic model.
        """
        if not response_text or not response_text.strip():
            logger.warning("Empty response received; returning default model")
            return model().dict()

        # Extract JSON substring from the response
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                validated = model(**data)
                return validated.dict()
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse/validate JSON: {e}")
                return model().dict()
        logger.warning("No JSON found in response; returning default model")
        return model().dict()


gemini_client = GeminiClient()
