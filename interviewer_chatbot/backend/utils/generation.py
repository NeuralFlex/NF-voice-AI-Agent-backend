import logging
import textwrap
from typing import Any, Dict
import json
from services.gemini_client import gemini_client
from utils.prompt_template import safe_prompt

logger = logging.getLogger(__name__)


def safe_text(text: str, max_len: int = 2000) -> str:
    """
    Sanitize and truncate user-provided or large text to avoid context overflow
    and unwanted characters.
    """
    if not text:
        return ""
    sanitized = str(text).replace("\r", "").replace("\t", "    ")
    return sanitized[:max_len]


def _safe_generate(prompt: str, fallback: str, gemini_client=gemini_client) -> str:
    try:
        return gemini_client.generate_content(prompt) or fallback
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return fallback


def safe_parse_json(response: Any) -> Dict[str, Any]:
    fallback = {"rating": 6, "feedback": "Good effort. Could elaborate more."}

    if not response:
        return fallback

    if isinstance(response, dict):
        return response

    text = None
    if hasattr(response, "text"):
        text = response.text
    elif hasattr(response, "content"):
        text = response.content
    elif isinstance(response, str):
        text = response

    if not text:
        return fallback

    try:
        return json.loads(text)
    except Exception:
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return {**fallback, "raw_text": text[:500]}


def build_prompt(role_desc: str, content: str, body: str) -> str:
    """
    Standard prompt builder to avoid duplication.
    Applies safe_text to content and strips extra whitespace.
    """
    return f"""
        You are {role_desc}.
        Using the following reference content:
        {safe_text(content)}

        {body}
    """.strip()
