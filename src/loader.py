import os
import sys
import logging
from pathlib import Path

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))

logger = logging.getLogger(__name__)


class TaxDocumentLoader:
    """
    Loads and chunks tax documents from a folder.
    Handles multiple PDFs and returns clean chunks.
    """

    def __init__(self, docs_folder: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.docs_folder = Path(docs_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        self._validate_folder()

    def _validate_folder(self) -> None:
        """Check folder exists and has PDFs."""
        if not self.docs_folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {self.docs_folder}"
            )
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(
                f"No PDFs found in: {self.docs_folder}"
            )
        logger.info(f"Found {len(pdf_files)} PDF(s)")

    def load(self) -> list:
        """Load all PDFs and return chunks with metadata."""
        all_chunks = []
        pdf_files = list(self.docs_folder.glob("*.pdf"))

        for pdf_path in pdf_files:
            logger.info(f"Loading: {pdf_path.name}")
            try:
                doc_loader = PyPDFLoader(str(pdf_path))
                pages = doc_loader.load()
                doc_chunks = self.splitter.split_documents(pages)

                for chunk in doc_chunks:
                    chunk.metadata["filename"] = pdf_path.name
                    chunk.metadata["source"] = str(pdf_path)

                all_chunks.extend(doc_chunks)
                logger.info(
                    f"  → {len(pages)} pages, "
                    f"{len(doc_chunks)} chunks"
                )

            except Exception as e:
                logger.error(f"ERROR loading {pdf_path.name}: {e}")
                continue

        logger.info(f"Total chunks: {len(all_chunks)}")
        return all_chunks

    def get_stats(self) -> dict:
        """Return loading statistics."""
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        return {
            "total_files": len(pdf_files),
            "filenames": [f.name for f in pdf_files],
            "docs_folder": str(self.docs_folder),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    DOCS_FOLDER = os.getenv("DOCS_FOLDER", "D:\\")

    print("Testing TaxDocumentLoader...")
    print(f"Loading PDFs from: {DOCS_FOLDER}\n")

    tax_loader = TaxDocumentLoader(
        docs_folder=DOCS_FOLDER,
        chunk_size=500,
        chunk_overlap=50
    )

    stats = tax_loader.get_stats()
    print(f"Files found: {stats['total_files']}")
    for name in stats['filenames']:
        print(f"  - {name}")

    print("\nLoading and chunking documents...")
    result_chunks = tax_loader.load()

    print(f"\nFirst chunk preview:")
    print(f"  File: {result_chunks[0].metadata['filename']}")
    print(f"  Page: {result_chunks[0].metadata.get('page', 'N/A')}")
    print(f"  Content: {result_chunks[0].page_content[:150]}...")
    print(f"\nDone! Total chunks: {len(result_chunks)}")