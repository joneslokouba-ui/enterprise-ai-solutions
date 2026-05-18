import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """
    Central configuration for the Tax RAG Assistant.
    Loads from .env and validates all required variables on startup.
    """

    # Groq settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    # Embedding settings — free local model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # Document settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # Paths
    DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "data/tax_docs")
    FAISS_INDEX_PATH: str = os.getenv(
        "FAISS_INDEX_PATH", "faiss_index"
    )

    # App settings
    APP_TITLE: str = "Tax RAG Assistant"
    APP_VERSION: str = "1.0.0"
    MAX_QUESTION_LENGTH: int = 1000

    @classmethod
    def validate(cls) -> None:
        """
        Validate all required config values exist.
        Fail fast at startup — never fail silently.
        """
        required = {
            "GROQ_API_KEY": cls.GROQ_API_KEY,
        }

        missing = [
            key for key, val in required.items() if not val
        ]

        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}\n"
                f"Please add them to your .env file."
            )

        logger.info(
            f"Config validated — {cls.APP_TITLE} v{cls.APP_VERSION}"
        )
        logger.info(f"Model: {cls.GROQ_MODEL}")
        logger.info(f"Docs folder: {cls.DOCS_FOLDER}")


if __name__ == "__main__":
    Config.validate()
    print("All configuration loaded successfully.")