import streamlit as st

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        # st.write(response) # We'll let app.py handle the display
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    else:
        st.session_state.messages.append({"role": "assistant", "content": "Please upload and process your PDFs first."})