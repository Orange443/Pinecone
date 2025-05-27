from importlib.metadata import files
import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
from pdf_text import get_pdf_text, divide_into_chunks, enrich_chunks, estimate_processing_time
from vectordb import embed_and_store,create_index, delete_index,query_pinecone,query_and_display_chunks,check_pinecone_index_exists
from pipeline import run_step1_chunking, run_step2_enrichment_optimized_async, run_step3_upsert
from QandA import handle_userinput, get_groq_response


st.set_page_config(
        page_title = "Chat with PDF",
        layout = "centered",
        page_icon="💬"
    )

def main():
    load_dotenv()
    st.title("Chat with PDFs")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

        if st.button("Delete Pinecone Index",type="secondary"):
            delete_index()
            st.session_state.messages = [{"role": "assistant", "content": "Index deleted. How can I help you?"}]
            # Rerun the app to reflect the changes immediately
            st.rerun()

        # --- Button to run the new pipeline ---
        if st.button("Process PDFs (Optimized)", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                st.info("Starting Optimized PDF Processing Pipeline...")
                create_index()

                step1_ok = run_step1_chunking(pdf_docs)

                step2_ok = False
                if step1_ok:
                    # --- Call the async function ---
                    try:
                        step2_ok = asyncio.run(run_step2_enrichment_optimized_async())
                    except Exception as e:
                        st.error(f"Error running async enrichment: {e}")
                        step2_ok = False
                    # --- End async call ---

                step3_ok = False
                if step2_ok:
                    step3_ok = run_step3_upsert()

                if step1_ok and step2_ok and step3_ok:
                    st.balloons()
                    st.success("🎉 Pipeline Complete!")
                else:
                    st.error("Pipeline failed. Check messages.")
        
        st.markdown("---")


    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if query_text := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query_text})
        st.chat_message("user").write(query_text)

        with st.spinner("Thinking..."):
            # Query Pinecone for relevant context
            if not check_pinecone_index_exists("llama-text-embed-v2"):
                # If index doesn't exist, show the message
                st.session_state.messages.append({"role": "assistant", "content": "The database index wasn't found. Please upload and process your PDFs first using the sidebar."})
                st.chat_message("assistant").write("The database index wasn't found. Please upload and process your PDFs first using the sidebar.")
            else: 
                response = query_pinecone(query_text) # <-- Call query_pinecone with query_text
                if response and response["matches"]:
                    context = " ".join([match.get("metadata", {}).get("text", "") for match in response["matches"]])
                    # Get response 
                    groq_response = get_groq_response(query_text, context)
                    st.session_state.messages.append({"role": "assistant", "content": groq_response})
                    st.chat_message("assistant").write(groq_response)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I couldn't find relevant information in your PDFs to answer that question."})
                    st.chat_message("assistant").write("I couldn't find relevant information in your PDFs to answer that question.")

if __name__ == '__main__':
    main()
