import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
import time
import os
from google.api_core import exceptions

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
    max_retries = 3

    for i, chunk in enumerate(chunks):
        prompt = (
            "Summarize the following text in 1-2 sentences and list 3-5 keywords.\n\n"
            f"{chunk}\n\n"
            "Respond exactly in this format:\n"
            "Description: <summary>\n"
            "Keywords: <comma-separated list>"
        )
        retries = 0
        while retries < max_retries:
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
                st.write(f"Processed chunk {i+1}...")
                break # Success!

            except exceptions.ResourceExhausted as e:
                st.warning(f"Rate limit hit on chunk {i+1}. Waiting 60s... (Attempt {retries+1}/{max_retries})")
                time.sleep(60)
                retries += 1
            except exceptions.ServiceUnavailable as e:
                st.warning(f"Gemini unavailable on chunk {i+1}. Waiting 10s... (Attempt {retries+1}/{max_retries})")
                time.sleep(10)
                retries += 1
            except Exception as e:
                st.error(f"Gemini failed on chunk {i+1} with unhandled error: {e}")
                enriched.append({"chunk": chunk, "description": "Error", "keywords": "Error"})
                break # Stop trying on unhandled errors
        if retries == max_retries:
             st.error(f"Failed to process chunk {i+1} after {max_retries} attempts.")
             enriched.append({"chunk": chunk, "description": "Failed", "keywords": "Failed"})

        # --- IMPORTANT: Pause *between* requests ---
        # Adjust sleep time based on your RPM limit (e.g., 30 RPM -> 60s/30 = 2s per request. Sleep slightly more.)
        time.sleep(4.5) # For 30 RPM limit, wait ~2.5s

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
