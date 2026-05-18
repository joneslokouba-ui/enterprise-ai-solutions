import os
import sys
import logging
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")


class TaxEmbedder:
    """
    Converts document chunks into vectors and stores
    them in a FAISS index for fast similarity search.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.index_path = Path(FAISS_INDEX_PATH)
        logger.info("Embedding model loaded successfully")

    def build_index(self, chunks: list) -> FAISS:
        """
        Convert chunks to vectors and build FAISS index.
        This runs once — saves index to disk for reuse.
        """
        logger.info(f"Building FAISS index from {len(chunks)} chunks...")
        logger.info("This may take a few minutes — please wait...")

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Save index to disk so we don't rebuild every time
        self.index_path.mkdir(exist_ok=True)
        vectorstore.save_local(str(self.index_path))

        logger.info(f"Index saved to: {self.index_path}")
        return vectorstore

    def load_index(self) -> FAISS:
        """
        Load existing FAISS index from disk.
        Much faster than rebuilding every time.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found at: {self.index_path}\n"
                f"Please run build_index() first."
            )

        logger.info(f"Loading existing index from: {self.index_path}")
        vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Index loaded successfully")
        return vectorstore

    def index_exists(self) -> bool:
        """Check if FAISS index already exists on disk."""
        return (self.index_path / "index.faiss").exists()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Import loader
    from src.loader import TaxDocumentLoader

    DOCS_FOLDER = os.getenv("DOCS_FOLDER", "D:\\")

    print("Testing TaxEmbedder...")
    print("Step 1: Loading documents...")

    tax_loader = TaxDocumentLoader(
        docs_folder=DOCS_FOLDER,
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = tax_loader.load()
    print(f"Loaded {len(chunks)} chunks\n")

    print("Step 2: Building embeddings and FAISS index...")
    print("(This takes 3-5 minutes for 3,797 chunks — normal!)\n")

    embedder = TaxEmbedder()

    if embedder.index_exists():
        print("Index already exists — loading from disk...")
        vectorstore = embedder.load_index()
    else:
        print("Building new index...")
        vectorstore = embedder.build_index(chunks)

    print("\nStep 3: Testing search...")
    test_question = "What is the standard deduction?"
    results = vectorstore.similarity_search(test_question, k=3)

    print(f"Question: '{test_question}'")
    print(f"Top 3 results found:\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  File: {doc.metadata.get('filename', 'Unknown')}")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Content: {doc.metadata['filename']}...")
        print(f"  Preview: {doc.page_content[:100]}...")
        print()

    print("Embedder test complete!")