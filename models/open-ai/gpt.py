from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.gpt4-all.xyz/v1")

def classify_sentence(text: str) -> str:
    """
    Classify the given sentence as 'safe', 'sexist', 'racist', or 'homophobic'.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an offensive sentence classifier. Respond only with one of these words: 'safe', 'sexist', 'racist', 'homophobic'."},
            {"role": "user", "content": f"Sentence: {text}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()