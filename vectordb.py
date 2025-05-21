
import os
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone 
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "llama-text-embed-v2"  
index = pc.Index(index_name)

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
    document_params={"input_type": "passage"}   
)

def vectorstore():
    pass




def embed_and_store(chunks):
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
