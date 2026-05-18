import azure.functions as func
import logging
import json
import os
from groq import Groq

# Initialize the Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route(route="tax-query", methods=["GET", "POST"])
def tax_query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Tax RAG API — Processing request')

    try:
        # Get question from request
        req_body = req.get_json()
        question = req_body.get('question', '')

        if not question:
            return func.HttpResponse(
                json.dumps({"error": "Please provide a question"}),
                mimetype="application/json",
                status_code=400
            )

        # Query Groq LLaMA
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful US Tax Assistant. Answer questions about US Federal Tax laws clearly and accurately."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response.choices[0].message.content

        return func.HttpResponse(
            json.dumps({
                "question": question,
                "answer": answer,
                "model": "llama-3.3-70b-versatile",
                "status": "success"
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )