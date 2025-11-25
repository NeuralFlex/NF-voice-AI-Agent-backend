import google.generativeai as genai
from config.settings import settings

genai.configure(api_key=settings.gemini_api_key)

GeminiModel = genai.GenerativeModel(settings.gemini_model)
