import streamlit as st
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec # For Pinecone
import time
import uuid # For generating unique IDs for vectors

# --- Placeholder Helper Functions (User should replace with their actual implementations) ---
def get_pdf_text(pdf_docs):
    """
    Placeholder: Extracts text from uploaded PDF files.
    Replace with your actual PDF text extraction logic.
    """
    all_text = []
    if pdf_docs:
        for pdf in pdf_docs:
            # In a real app, you'd use a library like PyMuPDF (fitz), pdfplumber, etc.
            all_text.append(f"Extracted text from '{pdf.name}': Content of the PDF...\n\n")
    if not all_text:
        return ""
    return "".join(all_text)

def divide_into_chunks(raw_text, chunk_size=1000, chunk_overlap=200):
    """
    Placeholder: Divides raw text into manageable chunks.
    Replace with your actual text chunking strategy (e.g., RecursiveCharacterTextSplitter from LangChain).
    """
    if not raw_text:
        return []
    
    # Simple sliding window chunking for demonstration
    chunks = []
    start = 0
    while start < len(raw_text):
        end = start + chunk_size
        chunks.append(raw_text[start:end])
        start += (chunk_size - chunk_overlap)
        if start >= len(raw_text): # ensure last part is not missed if overlap makes it skip
            break 
    # Add the very last bit if it was missed and is substantial
    if chunks and len(raw_text) > (len(chunks) * (chunk_size - chunk_overlap) - chunk_overlap): # Heuristic
        last_chunk_end = (len(chunks) -1) * (chunk_size - chunk_overlap) + chunk_size
        if last_chunk_end < len(raw_text):
             chunks.append(raw_text[last_chunk_end:])

    return [chunk for chunk in chunks if chunk.strip()]


# --- Placeholder Embedding Generation (CRITICAL: User MUST replace this) ---
# This is where you would integrate your "llama-text-embed-v2" model.
# The output for each chunk should be a list of floats (the vector).
# The dimension of this vector MUST match your Pinecone index's dimension.

# Replace this with the actual dimension of your "llama-text-embed-v2" model
LLAMA_EMBEDDING_DIMENSION = 768 # EXAMPLE DIMENSION - CHANGE THIS!

def generate_embeddings_for_chunks(text_chunks_list, model_name="llama-text-embed-v2"):
    """
    Placeholder function to generate embeddings for a list of text chunks.
    YOU MUST REPLACE THIS WITH YOUR ACTUAL EMBEDDING GENERATION LOGIC
    using the "llama-text-embed-v2" model you intend to use.

    For example, if you were using Sentence Transformers:
    ----------------------------------------------------
    from sentence_transformers import SentenceTransformer
    # Initialize the model (do this once, perhaps outside this function or cached)
    # model = SentenceTransformer('your-actual-llama-text-embed-v2-model-name-on-huggingface')
    # embeddings = model.encode(text_chunks_list).tolist()
    # return embeddings

    If you are using a class like the `PineconeEmbeddings` you mentioned:
    -------------------------------------------------------------------
    # Make sure this `PineconeEmbeddings` class has a method to embed multiple documents.
    # pinecone_embeddings_object = PineconeEmbeddings(
    #     model="llama-text-embed-v2",
    #     pinecone_api_key=os.environ["PINECONE_API_KEY"], # Or however it's configured
    #     document_params={"input_type": "passage"}
    # )
    # embeddings = pinecone_embeddings_object.embed_documents(text_chunks_list) # Method name might vary
    # return embeddings
    """
    st.info(f"Using PLACEHOLDER embedding generation for {len(text_chunks_list)} chunks.")
    if not text_chunks_list:
        return []

    dummy_embeddings = []
    for i, chunk in enumerate(text_chunks_list):
        # This creates a DUMMY embedding. Replace with REAL embeddings.
        dummy_vector = [(float(i + 1) * 0.01 + float(j) * 0.001) for j in range(LLAMA_EMBEDDING_DIMENSION)]
        # Normalize the dummy vector (cosine similarity often works better with normalized vectors)
        norm = sum(x*x for x in dummy_vector)**0.5
        normalized_dummy_vector = [x/norm if norm > 0 else 0.0 for x in dummy_vector]
        dummy_embeddings.append(normalized_dummy_vector)
        if i < 2: # Log first few for feedback
            st.write(f"  - Placeholder embedding for chunk {i+1} (first 3 dims): {dummy_embeddings[-1][:3]}...")
    
    st.success("Placeholder embedding generation complete.")
    return dummy_embeddings
# --- End Placeholder Functions ---

def main():
    st.set_page_config(
        page_title="Chat with PDF & Store Embeddings",
        layout="wide", # Changed to wide for better layout
        page_icon="💬"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    load_dotenv() # Load environment variables from .env file

    st.title("📄 Chat with your PDFs & Upload Embeddings to Pinecone 🌲")

    # Pinecone Configuration
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    INDEX_NAME = "your-pdf-embeddings-index" # IMPORTANT: Choose your Pinecone index name

    if not PINECONE_API_KEY:
        st.error("🔴 PINECONE_API_KEY not found in environment variables. Please set it in your .env file or system environment.")
        return

    # Initialize Pinecone connection
    # This can be cached in Streamlit for efficiency using @st.cache_resource
    @st.cache_resource
    def init_pinecone():
        try:
            pc_instance = Pinecone(api_key=PINECONE_API_KEY)
            return pc_instance
        except Exception as e_init:
            st.error(f"🔴 Failed to initialize Pinecone: {e_init}")
            return None
    
    pc = init_pinecone()
    if not pc:
        return

    col1, col2 = st.columns([1, 2]) # Sidebar-like column and main content column

    with col1:
        st.subheader("⚙️ PDF Processing")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, key="pdf_uploader")

        if st.button("🚀 Process PDFs & Upload to Pinecone", type="primary"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF file.")
                return

            with st.spinner("🔄 Processing PDFs, generating embeddings, and uploading to Pinecone... This may take a while."):
                # 1. Get PDF text
                st.info("Step 1: Extracting text from PDFs...")
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("🔴 Could not extract text from PDFs or PDFs are empty.")
                    return
                st.success("✅ Text extraction complete.")
                # st.text_area("Extracted Text (first 2000 chars):", raw_text[:2000], height=100, key="raw_text_display")

                # 2. Divide text into chunks
                st.info("Step 2: Dividing text into chunks...")
                chunks = divide_into_chunks(raw_text)
                if not chunks:
                    st.warning("⚠️ No text chunks were generated. The document might be too short or empty.")
                    return
                st.success(f"✅ Divided text into {len(chunks)} chunks.")
                # st.write(f"First chunk snippet: {chunks[0][:100]}...")

                # 3. Generate embeddings for chunks
                st.info(f"Step 3: Generating embeddings for {len(chunks)} chunks (using placeholder)...")
                # IMPORTANT: Replace `generate_embeddings_for_chunks` with your actual
                # "llama-text-embed-v2" embedding logic.
                # Ensure LLAMA_EMBEDDING_DIMENSION is correctly set.
                chunk_embeddings = generate_embeddings_for_chunks(chunks)
                
                if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
                    st.error("🔴 Embedding generation failed or mismatch in number of embeddings and chunks.")
                    return
                st.success(f"✅ Generated {len(chunk_embeddings)} embeddings.")

                # 4. Connect to or create Pinecone Index
                try:
                    st.info(f"Step 4: Connecting to Pinecone index: '{INDEX_NAME}'...")
                    if INDEX_NAME not in pc.list_indexes().names:
                        st.info(f"⏳ Index '{INDEX_NAME}' does not exist. Creating it...")
                        pc.create_index(
                            name=INDEX_NAME,
                            dimension=LLAMA_EMBEDDING_DIMENSION, # CRITICAL: Must match your embedding model's output dimension
                            metric="cosine", # Common for text embeddings (dotproduct or euclidean also options)
                            spec=ServerlessSpec(
                                cloud='aws',  # Or 'gcp', 'azure'. Check Pinecone docs for free tier availability & your setup
                                region='us-east-1' # Example region, choose one available for your Pinecone plan
                            )
                        )
                        # Wait for index to be ready (simple loop, consider timeout)
                        wait_time = 0
                        max_wait_time = 300 # 5 minutes
                        while not pc.describe_index(INDEX_NAME).status['ready']:
                            st.info(f"  Waiting for index to be ready... ({wait_time}s)")
                            time.sleep(5)
                            wait_time += 5
                            if wait_time > max_wait_time:
                                st.error("🔴 Index creation timed out.")
                                return
                        st.success(f"✅ Index '{INDEX_NAME}' created and ready.")
                    else:
                        st.success(f"✅ Successfully connected to existing index '{INDEX_NAME}'.")

                    index = pc.Index(INDEX_NAME)
                    initial_stats = index.describe_index_stats()
                    st.write(f"Index stats before upsert: {initial_stats}")

                except Exception as e_pinecone_index:
                    st.error(f"🔴 Pinecone index error: {e_pinecone_index}")
                    return

                # 5. Prepare vectors for upsert and upsert in batches
                st.info("Step 5: Preparing and upserting vectors to Pinecone...")
                vectors_to_upsert = []
                for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    vector_id = str(uuid.uuid4()) # Generate a unique ID for each vector
                    # Store the chunk text (or part of it) as metadata.
                    # Be mindful of Pinecone's metadata size limits (typically around 40KB per vector).
                    metadata = {"text_chunk": chunk_text[:10000]} # Store a good portion of the chunk
                    vectors_to_upsert.append({"id": vector_id, "values": embedding, "metadata": metadata})

                # Upsert in batches
                batch_size = 100 # Pinecone recommends batches of 100 or fewer for serverless, or up to 2MB payload.
                num_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size

                for i in range(num_batches):
                    batch_start = i * batch_size
                    batch_end = (i + 1) * batch_size
                    batch = vectors_to_upsert[batch_start:batch_end]
                    try:
                        st.info(f"  Upserting batch {i+1}/{num_batches} ({len(batch)} vectors)...")
                        index.upsert(vectors=batch) # Pass the list of dicts directly
                        st.success(f"  ✅ Batch {i+1} upserted.")
                    except Exception as e_upsert:
                        st.error(f"🔴 Error upserting batch {i+1}: {e_upsert}")
                        # Optionally, decide if you want to stop or continue with other batches
                
                st.success(f"🎉 All {len(vectors_to_upsert)} vectors processed and upsertion attempts made to '{INDEX_NAME}'.")
                st.balloons()
                final_stats = index.describe_index_stats()
                st.write(f"Index stats after upsert: {final_stats}")

    with col2:
        st.subheader("💬 Chat Interface")
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask something about your documents... (Chat logic TBD)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                # This is where you would:
                # 1. Generate embedding for the `prompt` (using the same model).
                # 2. Query the Pinecone `index` with this embedding (`index.query(...)`).
                # 3. Get relevant text chunks from query results' metadata.
                # 4. Pass the prompt and context chunks to an LLM to generate an answer.
                
                # For now, just a placeholder response:
                assistant_response = "The PDF content has been processed and embeddings uploaded to Pinecone. Querying and response generation logic is not yet implemented in this example."
                full_response = ""
                for char_resp in assistant_response: # Simulate stream
                    full_response += char_resp
                    time.sleep(0.01)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()
