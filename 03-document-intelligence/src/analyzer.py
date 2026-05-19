import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_document(text, question):
    """Answer questions about the document."""
    prompt = f"""Based on the following document content, answer this question:

Question: {question}

Document Content:
{text[:3000]}

Provide a clear and accurate answer based only on the document content."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful document analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def summarize_document(text):
    """Generate document summary."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful document summarizer."},
            {"role": "user", "content": f"Summarize this document:\n\n{text[:3000]}"}
        ]
    )
    return response.choices[0].message.content