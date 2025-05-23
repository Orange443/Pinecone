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
    document_params={"input_type": "passage"}   
)

def create_index():
    index_name = "llama-text-embed-v2"
    
    #existing_indexes = [idx.strip().lower() for idx in pc.list_indexes().names()]
    existing_indexes = pc.list_indexes().names()
    st.write("Available indexes:", existing_indexes)
    
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

def update_db():
    response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
    )
    testing = response.text
    return testing


def vectorstore_embed():
    pass
    # embeddings = pc.inference.embed(
    # model="multilingual-e5-large",
    # inputs=[c['text'] for c in chunks],
    # parameters={"input_type": "passage", "truncate": "END"}
    # )
    # print(embeddings[0])

def embed_and_store(chunks):
    index_name = "llama-text-embed-v2"
    index = pc.Index(index_name)
    try:
        vectors = embeddings.embed_documents(chunks)

        to_upsert = [
            {
                "id": f"chunk-{i}",
                "values": vectors[i],
                "metadata": {"text": chunks[i]}
            }
            for i in range(len(chunks))
        ]
        batch_size = 100  
        for i in range(0, len(to_upsert), batch_size):
            batch = to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        
        return len(to_upsert)
    except Exception as e:
        print(f"An error occurred during embedding or upserting: {e}")
        return 0 

def get_pinecone_vectorstore():
    index_name = "llama-text-embed-v2"
    index = pc.Index(index_name)
    
    query_embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2", 
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        query_params={"input_type": "query"} 
    )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,  
        embedding=query_embeddings
    )
    return vectorstore

def get_converstation_chain(vectorstore):
    
    llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


