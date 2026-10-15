
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY backend/.env mein nahi mili.")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemma-4-26b-a4b-it",
            contents="Introduce yourself in one short sentence."
        )

        print("SUCCESS: Gemma API connected!")
        print("Model response:", response.text)

    except Exception as e:
        print("API test failed.")
        print(type(e).__name__, str(e))
