import os
import sys
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

TAX_PROMPT_TEMPLATE = """You are a professional tax assistant with
expertise in US federal tax law. Answer tax questions accurately
using ONLY the context provided below.

Rules:
1. Answer ONLY from the context provided
2. Always cite the source document and page number
3. If answer not in context say:
   "I could not find this in the provided tax documents.
   Please consult a qualified tax professional."
4. Never make up tax figures, rates or rules
5. Keep answers clear and professional

Context from tax documents:
{context}

Question: {question}

Answer (with citations):"""

TAX_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=TAX_PROMPT_TEMPLATE
)


def format_docs(docs: list) -> str:
    """Format retrieved chunks into a single context string."""
    formatted = []
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Source: {filename}, Page {page}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)


class TaxRetriever:
    """
    Connects FAISS vector store to Groq LLM.
    Retrieves relevant chunks and generates cited answers.
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": TOP_K_RESULTS}
        )

        logger.info(f"Connecting to Groq: {GROQ_MODEL}")
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=1024
        )

        # Modern LangChain LCEL chain — replaces RetrievalQA
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | TAX_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info("Tax retriever ready")

    def ask(self, question: str) -> dict:
        """
        Ask a tax question and get a cited answer.
        Returns answer text and source documents.
        """
        if not question or not question.strip():
            return {
                "answer": "Please enter a valid question.",
                "sources": [],
                "error": None
            }

        if len(question) > 1000:
            return {
                "answer": "Question too long. Max 1000 characters.",
                "sources": [],
                "error": None
            }

        try:
            logger.info(f"Processing: {question[:50]}...")

            # Get answer from chain
            answer = self.chain.invoke(question)

            # Get source documents separately
            source_docs = self.retriever.invoke(question)

            # Extract unique sources
            sources = []
            seen = set()
            for doc in source_docs:
                filename = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page", "N/A")
                source_key = f"{filename}_p{page}"
                if source_key not in seen:
                    sources.append({
                        "filename": filename,
                        "page": page,
                        "preview": doc.page_content[:150]
                    })
                    seen.add(source_key)

            return {
                "answer": answer,
                "sources": sources,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": None,
                "sources": [],
                "error": str(e)
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from embedder import TaxEmbedder

    print("Testing TaxRetriever...")
    print("Loading FAISS index...\n")

    tax_embedder = TaxEmbedder()
    tax_vectorstore = tax_embedder.load_index()

    tax_retriever = TaxRetriever(tax_vectorstore)

    # Test questions
    test_questions = [
        "What is the standard deduction for 2024?",
        "How do I report freelance income?",
        "What are the tax brackets for single filers?"
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        print("-" * 50)
        response = tax_retriever.ask(q)

        if response["error"]:
            print(f"Error: {response['error']}")
        else:
            print(f"Answer:\n{response['answer']}")
            print(f"\nSources:")
            for source in response["sources"]:
                print(
                    f"  - {source['filename']} "
                    f"(Page {source['page']})"
                )
        print()

    print("Retriever test complete!")