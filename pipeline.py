# pipeline.py (Optimized Version)
import os
import pandas as pd
import time
import datetime
import asyncio 
import json     
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
from google.api_core import exceptions
from dotenv import load_dotenv
from vectordb import pc, embeddings
import streamlit as st

load_dotenv()

# --- Configuration ---
CHUNK_CSV_PATH = "chunks.csv"
ENRICHED_CSV_PATH = "enriched_chunks.csv"
INDEX_NAME = "llama-text-embed-v2"
GEMINI_MODEL = "gemini-1.5-flash-latest"
EMBEDDING_MODEL = "llama-text-embed-v2"
PINECONE_BATCH_SIZE = 100

# --- Optimization Config ---
BATCH_SIZE = 5       
CONCURRENT_LIMIT = 30 # Max parallel requests (Free Tier RPM)

# ... (Keep your client initializations) ...
google_api_key = os.environ['GOOGLE_API_KEY']
genai_client = genai.Client(api_key=google_api_key)
# ... (Keep Pinecone initializations) ...

# ... (Keep run_step1_chunking and other functions) ...
def get_pdf_text_streamlit(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not read {pdf.name}: {e}")
    return text

def divide_into_chunks_pipeline(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len,
        separators=["\n\n", "\n", " ", "", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002"]
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def run_step1_chunking(pdf_docs):
    st.info("Step 1: Reading and chunking PDFs...")
    raw_text = get_pdf_text_streamlit(pdf_docs)
    if not raw_text: st.error("Step 1 Failed..."); return False
    chunks = divide_into_chunks_pipeline(raw_text)
    if not chunks: st.warning("Step 1 Warning..."); return False
    all_chunks_data = [{"chunk_id": f"chunk_{i}", "chunk_text": chunk} for i, chunk in enumerate(chunks)]
    df = pd.DataFrame(all_chunks_data)
    df.to_csv(CHUNK_CSV_PATH, index=False, encoding='utf-8')
    st.success(f"Step 1: Saved {len(df)} chunks to {CHUNK_CSV_PATH}")
    return True

# --- New Async/Batch Enrichment Functions ---

def build_batch_prompt(batch):
    """Builds a prompt to process multiple chunks and return JSON."""
    prompt_header = (
        "Process each of the following text chunks individually. "
        "For each chunk, provide a 1-2 sentence summary ('description') and 3-5 keywords ('keywords'). "
        "Return the output *only* as a valid JSON list, where each item has 'chunk_id', 'description', and 'keywords' keys. "
        "Ensure the JSON is well-formed.\n\n"
        "Input Chunks:\n"
    )
    json_input = json.dumps([{"chunk_id": item['chunk_id'], "text": item['chunk_text']} for item in batch], indent=2)
    return prompt_header + json_input

def parse_gemini_json_response(text_response):
    """Tries to parse the JSON response from Gemini, handling potential issues."""
    try:
        # Sometimes Gemini wraps JSON in ```json ... ```
        if text_response.strip().startswith("```json"):
            text_response = text_response.strip()[7:-3].strip()
        return json.loads(text_response)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}\nResponse was:\n{text_response[:500]}...")
        return None # Indicate failure

async def enrich_batch_async(batch, client, semaphore):
    """Async version to enrich a BATCH of chunks."""
    async with semaphore:
        prompt = build_batch_prompt(batch)
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                loop = asyncio.get_running_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                )

                if hasattr(resp, 'text') and resp.text:
                    parsed_json = parse_gemini_json_response(resp.text)
                    if parsed_json and isinstance(parsed_json, list):
                         # Add original text back for CSV - match by ID
                        text_map = {item['chunk_id']: item['chunk_text'] for item in batch}
                        for item in parsed_json:
                            item['chunk_text'] = text_map.get(item['chunk_id'], "N/A")
                        return parsed_json # Returns a list of dicts
                    else:
                        print("Failed to parse or invalid JSON format.")
                        # Mark all in batch as failed
                        return [{"chunk_id": item['chunk_id'], "chunk_text": item['chunk_text'], "description": "Parse Error", "keywords": "Parse Error"} for item in batch]
                else:
                    print("Empty response from Gemini.")
                    return [{"chunk_id": item['chunk_id'], "chunk_text": item['chunk_text'], "description": "Empty Resp", "keywords": "Empty Resp"} for item in batch]

            except exceptions.ResourceExhausted:
                wait_time = 60 * (2**retries)
                print(f"Rate limit hit, waiting {wait_time}s... (Attempt {retries+1})")
                await asyncio.sleep(wait_time)
                retries += 1
            except Exception as e:
                print(f"Unhandled Error: {e}")
                await asyncio.sleep(5) # Small sleep on other errors
                return [{"chunk_id": item['chunk_id'], "chunk_text": item['chunk_text'], "description": "Error", "keywords": "Error"} for item in batch]

        print(f"Failed batch after {max_retries} retries.")
        return [{"chunk_id": item['chunk_id'], "chunk_text": item['chunk_text'], "description": "Failed", "keywords": "Failed"} for item in batch]

async def run_step2_enrichment_optimized_async():
    """Optimized async Step 2: Batches chunks and processes concurrently."""
    st.info("Step 2 (Optimized): Starting enrichment...")
    if not os.path.exists(CHUNK_CSV_PATH):
        st.error(f"Step 2 Failed: {CHUNK_CSV_PATH} not found!")
        return False

    df = pd.read_csv(CHUNK_CSV_PATH)
    chunks_to_process = df.to_dict('records')
    batches = [chunks_to_process[i:i + BATCH_SIZE] for i in range(0, len(chunks_to_process), BATCH_SIZE)]

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY']) # Re-init client for async context if needed

    tasks = [enrich_batch_async(batch, client, semaphore) for batch in batches]
    
    all_enriched_results = []
    progress_bar = st.progress(0, text="Enriching batches...")
    total_batches = len(tasks)
    start_time = time.time()

    for i, task in enumerate(asyncio.as_completed(tasks)):
        batch_result = await task
        all_enriched_results.extend(batch_result) # Add all results from the batch

        # Update progress based on batches
        eta_str = calculate_eta_text(i, total_batches, start_time) # Use your ETR func
        progress_text = f"Enriching batch {i + 1}/{total_batches}{eta_str}"
        progress_bar.progress((i + 1) / total_batches, text=progress_text)

    # Save all results
    pd.DataFrame(all_enriched_results).to_csv(ENRICHED_CSV_PATH, index=False, encoding='utf-8')
    st.success(f"Step 2: Saved {len(all_enriched_results)} enriched chunks to {ENRICHED_CSV_PATH}")
    return True

# --- Keep/Add your ETR function ---
def calculate_eta_text(current_chunk_index, total_chunks, start_time, start_index=0):
    chunks_processed = current_chunk_index + 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1 and chunks_processed > start_index:
        chunks_this_run = chunks_processed - start_index
        if chunks_this_run > 0:
            avg_time_per_chunk = elapsed_time / chunks_this_run
            remaining_chunks = total_chunks - chunks_processed
            eta_seconds = remaining_chunks * avg_time_per_chunk
            eta_formatted = str(datetime.timedelta(seconds=int(eta_seconds)))
            return f" (Est. {eta_formatted} left)"
    return " (Calculating...)"

# ... (Keep run_step3_upsert) ...
# pipeline.py (Modified run_step3_upsert function)

def run_step3_upsert():
    st.info("Step 3: Embedding and upserting to Pinecone...")
    if not pc or not embeddings:
        st.error("Step 3 Failed: Pinecone or Embeddings not initialized.")
        return False
    if not os.path.exists(ENRICHED_CSV_PATH):
        st.error(f"Step 3 Failed: Input file {ENRICHED_CSV_PATH} not found!")
        return False

    try:
        df = pd.read_csv(ENRICHED_CSV_PATH)
    except pd.errors.EmptyDataError:
        st.error(f"Step 3 Failed: {ENRICHED_CSV_PATH} is empty.")
        return False

    # Filter out rows that failed or had errors, and handle potential NaNs
    df = df[~df['description'].isin(['Failed', 'Error', 'Empty Resp', 'Parse Error', 'N/A'])].dropna()

    # --- ADD THIS LINE ---
    df = df.reset_index(drop=True)
    # --- END ADD ---

    if df.empty:
        st.warning("Step 3 Warning: No valid chunks found to embed after filtering.")
        return True # Not a failure, just nothing to do

    index = pc.Index(INDEX_NAME)
    to_embed = df['chunk_text'].tolist()
    st.info(f"Embedding {len(to_embed)} documents...")
    vectors = embeddings.embed_documents(to_embed)

    # Check if number of vectors matches DataFrame rows AFTER reset
    if len(vectors) != len(df):
        st.error(f"Step 3 Failed: Mismatch between DataFrame rows ({len(df)}) and generated vectors ({len(vectors)}).")
        return False

    all_upserts = []
    # Now, 'i' will correctly go from 0 to len(df)-1
    for i, row in df.iterrows():
        all_upserts.append({
            "id": row['chunk_id'],
            "values": vectors[i], # This will now work correctly
            "metadata": {
                "text": row["chunk_text"],
                "description": str(row["description"]), # Ensure metadata is string
                "keywords": str(row["keywords"])      # Ensure metadata is string
            }
        })

    total_upserted = 0
    st_upsert_progress = st.empty()
    for start in range(0, len(all_upserts), PINECONE_BATCH_SIZE):
        end = start + PINECONE_BATCH_SIZE
        batch = all_upserts[start:end]
        try:
            index.upsert(vectors=batch)
            total_upserted += len(batch)
            st_upsert_progress.info(f"Upserted {total_upserted}/{len(all_upserts)} vectors...")
        except Exception as e:
            st.error(f"Step 3 Failed: Error upserting batch: {e}")
            return False

    st.success(f"Step 3: Finished. Upserted {total_upserted} vectors.")
    return True


