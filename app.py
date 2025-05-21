from importlib.metadata import files
import streamlit as st
import os
from dotenv import load_dotenv
from pdf_text import get_pdf_text, divide_into_chunks
from vectordb import embed_and_store,get_pinecone_vectorstore,get_converstation_chain
from QandA import handle_userinput

def main():
    st.set_page_config(
        page_title = "Chat with PDF",
        layout = "centered",
        page_icon="💬"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    load_dotenv()

    st.title("HI")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner(""):
                #get pdf text
                raw_text =  get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #divide text into chunks 
                chunks = divide_into_chunks(raw_text)
                st.write(chunks)
                #num_chunks = embed_and_store(chunks)
                #st.success(f"Uploaded {num_chunks} chunks to Pinecone!")

                #vectorstore = get_pinecone_vectorstore()
                # """ st.session_state.conversation = get_converstation_chain(vectorstore)
                # st.success("PDFs processed and conversation chain ready!")
                # st.write(chunks) """


if __name__ == '__main__':
    main()
