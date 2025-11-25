from chromadb.api.models.Collection import Collection
import chromadb
from models.embedding_model import embeddings
import logging
from typing import Optional
from langchain_core.documents import Document
import os
from config.settings import settings

logger = logging.getLogger(__name__)

CHROMA_API_KEY = settings.chroma_api_key
CHROMA_TENANT = settings.chroma_tenant
CHROMA_DATABASE = settings.chroma_database

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)


def create_vectorstore(
    documents: list[Document], user_id: str = "default_user"
) -> Optional[Collection]:
    if not documents:
        logger.warning("No documents provided")
        return None

    collection_name = f"interviewer-chatbot-{user_id}"
    collection = client.get_or_create_collection(collection_name)

    doc_texts = [doc.page_content for doc in documents]
    doc_embeddings = [embeddings.embed_query(text) for text in doc_texts]
    doc_ids = [f"{user_id}_{i}" for i in range(len(documents))]

    collection.add(ids=doc_ids, documents=doc_texts, embeddings=doc_embeddings)
    return collection


def load_vectorstore(user_id: str = "default_user") -> Optional[Collection]:
    collection_name = f"interviewer-chatbot-{user_id}"
    return client.get_or_create_collection(collection_name)


def delete_vectorstore(user_id: str = "default_user") -> bool:
    """
    Delete a user's Chroma Cloud collection.

    Args:
        user_id (str): Identifier for the user.

    Returns:
        bool: True if deleted successfully, False otherwise.
    """
    try:
        collection_name = f"interviewer-chatbot-{user_id}"
        client.delete_collection(collection_name)
        logger.info("Deleted Chroma Cloud collection: %s", collection_name)
        return True
    except Exception as e:
        logger.error("Failed to delete vectorstore: %s", e, exc_info=True)
        return False
