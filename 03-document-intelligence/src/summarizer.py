from src.analyzer import client


def extract_key_points(text):
    """Extract key points from document."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful document analyst."
            },
            {
                "role": "user",
                "content": f"""Extract the 5 most important key points 
                from this document:\n\n{text[:3000]}

                Format as numbered list."""
            }
        ]
    )
    return response.choices[0].message.content