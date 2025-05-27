import os
import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from google import genai

load_dotenv()

# Initialize Pinecone with new SDK
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)

# Define embedding model
embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    pinecone_api_key=api_key,
    document_params={"input_type": "passage"},
    query_params={"input_type": "query"}  
)

def create_index():
    index_name = "llama-text-embed-v2"
    
    #existing_indexes = [idx.strip().lower() for idx in pc.list_indexes().names()]
    existing_indexes = pc.list_indexes().names()
    #st.write("Available indexes:", existing_indexes)
    
    if index_name not in existing_indexes:
        try:
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            st.success(f"Index '{index_name}' created.")
        except Exception as e:
            st.error(f"Error creating index: {e}")
    else:
        st.warning(f"Index '{index_name}' already exists.")

def delete_index():
    try:
        pc.delete_index("llama-text-embed-v2")
        st.success("Pinecone index 'llama-text-embed-v2' deleted successfully.")
    except Exception as e:
        st.error(f"Failed to delete index: {e}")


def embed_and_store(enriched_chunks: list[dict]) -> int:
    """
    Embeds only `chunk` text but upserts Description & Keywords as metadata.
    Returns number of vectors written.
    """
    index_name = "llama-text-embed-v2"
    index = pc.Index(index_name)

    try:
        # 1. vectors for text only
        to_embed = [c["chunk"] for c in enriched_chunks]
        vectors   = embeddings.embed_documents(to_embed)

        # 2. package for Pinecone
        upserts = []
        for i, entry in enumerate(enriched_chunks):
            upserts.append(
                {
                    "id": f"chunk-{i}",
                    "values": vectors[i],
                    "metadata": {
                        "text": entry["chunk"],
                        "description": entry["description"],
                        "keywords": entry["keywords"]
                    },
                }
            )

        # 3. batch upsert
        batch_size = 100
        for start in range(0, len(upserts), batch_size):
            index.upsert(vectors=upserts[start : start + batch_size])

        return len(upserts)

    except Exception as e:
        st.error(f"Embedding / upsert error: {e}")
        return 0

import streamlit as st

def query_pinecone(query_text, top_k=3):
    # Embed the query using the embed_query method
    query_embedding = embeddings.embed_query(query_text)

    # Access the Pinecone index
    index = pc.Index("llama-text-embed-v2")  # Ensure this matches your actual index name

    # Perform the query
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return response

def query_and_display_chunks():
    st.subheader("Query Pinecone")

    user_query = st.text_input("Ask a question or enter a keyword:")

    if user_query and st.button("Search"):
        with st.spinner("Searching Pinecone..."):
            try:
                # Embed the query
                query_embedding = embeddings.embed_query(user_query)

                # Access the Pinecone index
                index = pc.Index("llama-text-embed-v2") #---> Index name

                # Query the index
                response = index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True
                )

                # Display results
                if response["matches"]:
                    st.success(f"Found {len(response['matches'])} relevant chunks:")
                    for match in response["matches"]:
                        metadata = match.get("metadata", {})
                        st.markdown("**Chunk:**")
                        st.write(metadata.get("text", "No text found"))

                        # Optional metadata display
                        if "description" in metadata:
                            st.markdown(f"**Description:** {metadata['description']}")
                        if "keywords" in metadata:
                            st.markdown(f"**Keywords:** {metadata['keywords']}")
                        st.markdown("---")
                else:
                    st.warning("No matches found.")

            except Exception as e:
                st.error(f"An error occurred: {e}")



def check_pinecone_index_exists(index_name: str) -> bool:
    """
    Checks if a Pinecone index exists.

    Args:
        index_name: The name of the Pinecone index to check.

    Returns:
        True if the index exists, False otherwise.
    """
    try:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            st.error("PINECONE_API_KEY not found. Please set it.")
            return False

        pc = Pinecone(api_key=api_key)
        return index_name in pc.list_indexes().names()

    except Exception as e:
        st.error(f"Error checking Pinecone index: {e}")
        return False

