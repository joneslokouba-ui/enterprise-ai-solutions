import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def research_agent(topic):
    """Agent 1: Researches the given topic."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a research agent. Research the given topic thoroughly."
            },
            {
                "role": "user",
                "content": f"Research this topic: {topic}"
            }
        ]
    )
    return response.choices[0].message.content

def analysis_agent(research):
    """Agent 2: Analyzes the research."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an analysis agent. Analyze the research provided."
            },
            {
                "role": "user",
                "content": f"Analyze this research:\n\n{research}"
            }
        ]
    )
    return response.choices[0].message.content

def summary_agent(analysis):
    """Agent 3: Summarizes the analysis."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a summary agent. Create a clear concise summary."
            },
            {
                "role": "user",
                "content": f"Summarize this analysis:\n\n{analysis}"
            }
        ]
    )
    return response.choices[0].message.content