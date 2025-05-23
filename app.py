from importlib.metadata import files
import streamlit as st
import os
from dotenv import load_dotenv
from pdf_text import get_pdf_text, divide_into_chunks, enrich_chunks
from vectordb import embed_and_store,create_index, delete_index,query_pinecone,query_and_display_chunks
from QandA import handle_userinput

def main():
    st.set_page_config(
        page_title = "Chat with PDF",
        layout = "centered",
        page_icon="💬"
    )

    load_dotenv()
    st.title("Chat with PDFs")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)


        if st.button("Delete Pinecone Index",type="secondary"):
            delete_index()

        if st.button("Process PDFs",type="primary"):
            with st.spinner(""):
                #Step 1: Get raw text
                raw_text =  get_pdf_text(pdf_docs)

                #Step 2: divide text into chunks  
                chunks = divide_into_chunks(raw_text)
                #step 3: calls an API to make a structured format
                enriched_chunks = enrich_chunks(chunks)  
                st.write(enriched_chunks)
                #num_chunks = embed_and_store(chunks)
                #st.success(f"Uploaded {num_chunks} chunks to Pinecone!")
                create_index()
                num_chunks = embed_and_store(enriched_chunks)
                st.success(f"Uploaded {num_chunks} enriched chunks to Pinecone!")



                #vectorstore = get_pinecone_vectorstore()
                # """ st.session_state.conversation = get_converstation_chain(vectorstore)
                # st.success("PDFs processed and conversation chain ready!")
                # st.write(chunks) """"   

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    query_text = st.chat_input()
    query_pinecone()

if __name__ == '__main__':
    main()
