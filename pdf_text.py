import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
import os

client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

# ── helpers.py  (or wherever you keep helpers) ──────────────────────────
def enrich_chunks(chunks: list[str]) -> list[dict]:
    """
    For each chunk ask Gemini-Flash for a 1-2 sentence description and 3-5 keywords.
    Returns a list of dicts:
        { "chunk": <raw text>,
          "description": <summary>,
          "keywords": <comma-sep keywords> }
    """
    enriched = []

    for i, chunk in enumerate(chunks):
        prompt = (
            "Summarize the following text in 1–2 sentences and list 3-5 keywords.\n\n"
            f"{chunk}\n\n"
            "Respond exactly in this format:\n"
            "Description: <summary>\n"
            "Keywords: <comma-separated list>"
        )

        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            # --- very simple parse ---
            desc, kw = "", ""
            for line in resp.text.splitlines():
                if line.lower().startswith("description"):
                    desc = line.split(":", 1)[-1].strip()
                elif line.lower().startswith("keywords"):
                    kw = line.split(":", 1)[-1].strip()

            enriched.append(
                {"chunk": chunk, "description": desc, "keywords": kw}
            )

        except Exception as e:
            st.warning(f"Gemini failed on chunk {i}: {e}")
            enriched.append({"chunk": chunk, "description": "", "keywords": ""})

    return enriched

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def divide_into_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=[
        "\n\n",
        "\n",
        " ", 
        "",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",
        ]
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
