import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def get_groq_response(question, context):
    """
    Generates a response from Groq based on the question and context.
    """
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found. Please set it in your .env file.")
            return "Error: GROQ_API_KEY not set."

        # Choose a model, e.g., 'llama3-8b-8192' or 'mixtral-8x7b-32768'
        chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b")

        system = "You are a helpful assistant. Answer the following question based only on the provided context. If the context doesn't contain the answer, say 'I cannot answer this based on the provided documents.'"
        human = """
        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        chain = prompt | chat | StrOutputParser()

        response = chain.invoke({"context": context, "question": question})
        return response

    except Exception as e:
        st.error(f"Error generating response from Groq: {e}")
        return f"An error occurred: {e}"