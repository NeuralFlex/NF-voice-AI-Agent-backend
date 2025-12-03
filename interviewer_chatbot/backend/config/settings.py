import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://interview_user:postgres@localhost:5432/interview_db",
)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

gemini_model = os.getenv("GEMINI_MODEL")
gemini_embedding_model = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables!")


class Settings:
    def __init__(self):
        self.gemini_api_key = GEMINI_API_KEY
        self.tavily_api_key = TAVILY_API_KEY
        self.gemini_model = gemini_model
        self.gemini_embedding_model = gemini_embedding_model
        self.database_url = DB_URI
        self.chroma_api_key = CHROMA_API_KEY
        self.chroma_tenant = CHROMA_TENANT
        self.chroma_database = CHROMA_DATABASE
        self.livekit_api_key = LIVEKIT_API_KEY
        self.livekit_url = LIVEKIT_URL
        self.livekit_api_secret = LIVEKIT_API_SECRET


settings = Settings()
