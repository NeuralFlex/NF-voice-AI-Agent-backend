from tavily import TavilyClient
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TavilyService:
    """
    Wrapper for Tavily search API with logging and safe output.
    """

    def __init__(self):
        self.client = TavilyClient(api_key=settings.tavily_api_key)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """
        Perform a search query using Tavily and return a list of text snippets.
        """
        try:
            response = self.client.search(query=query, top_k=top_k)
            results = response.get("results", [])
            snippets = [
                r.get("snippet") or r.get("content") or r.get("title", "")
                for r in results
            ][:top_k]
            logger.info(
                f"Tavily search successful: {len(snippets)} results for query '{query}'"
            )
            return snippets
        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {e}")
            return []


tavily_service = TavilyService()
