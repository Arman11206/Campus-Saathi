
from pathlib import Path
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai

# ---------- CONFIGURATION ----------

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

load_dotenv(BACKEND_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemma-4-26b-a4b-it"

app = FastAPI(title="Campus Saathi API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=API_KEY) if API_KEY else None


class ChatRequest(BaseModel):
    question: str


# ---------- LOAD DOCUMENTS ONCE ----------

def load_documents():
    documents = []

    if not DOCUMENTS_DIR.exists():
        print(f"Documents folder not found: {DOCUMENTS_DIR}")
        return documents

    for file_path in DOCUMENTS_DIR.iterdir():
        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8")

                if text.strip():
                    documents.append({
                        "source": file_path.name,
                        "page": None,
                        "text": text,
                    })

            elif file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))

                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""

                    if text.strip():
                        documents.append({
                            "source": file_path.name,
                            "page": page_number,
                            "text": text,
                        })

        except Exception as error:
            print(f"Document error ({file_path.name}): {error}")

    print(f"Loaded {len(documents)} document sections.")
    return documents


DOCUMENTS_CACHE = load_documents()


# ---------- API ROUTES ----------

@app.get("/")
def home():
    return {
        "message": "Campus Saathi API is running!",
        "documents_loaded": len(DOCUMENTS_CACHE),
    }


@app.get("/documents")
def list_documents():
    return {
        "documents_loaded": len(DOCUMENTS_CACHE),
        "sources": [
            {"file": item["source"], "page": item["page"]}
            for item in DOCUMENTS_CACHE
        ],
    }


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="API key missing. Check backend/.env"
        )

    if not DOCUMENTS_CACHE:
        return {
            "answer": "No readable university documents found. Check the documents folder."
        }

    # Prepare a limited amount of reference text.
    context_parts = []
    total_chars = 0
    max_chars = 12000

    for item in DOCUMENTS_CACHE:
        remaining = max_chars - total_chars

        if remaining <= 0:
            break

        label = f"Source: {item['source']}"
        if item["page"] is not None:
            label += f", page {item['page']}"

        section = f"{label}\n{item['text']}"[:remaining]
        context_parts.append(section)
        total_chars += len(section)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are Campus Saathi, an assistant for Integral University students.

Answer the student's question clearly and concisely using the
university reference text below.

Never invent university facts, fees, dates, or policies.
If the answer is not supported by the text, say that you could not
verify it from the available documents and advise contacting the
university office.

Mention a source filename and page only when you can identify it
from the reference text. Never invent page numbers.

REFERENCE DOCUMENTS:
{context}

STUDENT QUESTION:
{question}

Give a direct, short answer.
"""

    try:
        print(f"Sending question to model: {MODEL_NAME}")

        # Start without restrictive generation settings while debugging.
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        # First try the SDK's standard text accessor.
        answer = response.text

        # Fallback: extract text from candidate parts if needed.
        if not answer and response.candidates:
            extracted_parts = []

            for candidate in response.candidates:
                content = candidate.content

                if content and content.parts:
                    for part in content.parts:
                        if getattr(part, "text", None):
                            extracted_parts.append(part.text)

            answer = "\n".join(extracted_parts).strip()

        if not answer:
            # Print diagnostic information in the backend terminal.
            print("MODEL RETURNED NO TEXT.")
            print("Response details:", response)

            if response.candidates:
                for candidate in response.candidates:
                    print("Finish reason:", candidate.finish_reason)

            raise HTTPException(
                status_code=502,
                detail="The model returned no text. Check the VS Code backend terminal."
            )

        print("Answer generated successfully.")
        return {"answer": answer}

    except HTTPException:
        raise

    except Exception as error:
        # Keep the real error in the terminal, not the API key.
        print(f"Gemma API ERROR: {type(error).__name__}: {error}")

        raise HTTPException(
            status_code=502,
            detail=f"AI request failed ({type(error).__name__}). Check the VS Code terminal."
        )
