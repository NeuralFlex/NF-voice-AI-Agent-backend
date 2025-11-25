import fitz
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF (fitz).

    Args:
        pdf_bytes (bytes): PDF file content in bytes.

    Returns:
        str: Extracted plain text from the PDF.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


def chunk_cv_text(cv_text: str, user_id: str = "default_user") -> list:
    """
    Splits CV text into chunks for embedding and retrieval.

    Args:
        cv_text (str): The raw CV text.
        user_id (str): Optional user identifier for metadata.

    Returns:
        List[Document]: List of Document objects with chunked CV text.
    """
    try:
        chunk_size = 800
        chunk_overlap = 200
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        cv_text = cv_text.strip().replace("\n", " ")
        documents = []

        for i, chunk in enumerate(splitter.split_text(cv_text)):
            chunk_text = chunk.strip()
            if len(chunk_text) < 20:
                continue
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={"user_id": user_id, "chunk_index": i},
                )
            )
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to chunk CV text: {e}")
