import streamlit as st
from dotenv import load_dotenv
import os

from functions import (LimitedSizeList,
                       get_texts_from_files,
                       split_text_into_chunks,
                       get_vectorstore,
                       get_embeddings,
                       get_llm, 
                       get_all_docs_embedding)

def main():

    st.set_page_config("Chat PDF")

    st.header("Your documents")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    model_id = st.selectbox(
        "model id", 
        options=[
            "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", 
            "gemini-1.5-pro", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", 
            "mistral-saba-24b", "qwen-2.5-coder-32b", "qwen-2.5-32b", "llama-3.3-70b-specdec", 
            "llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-11b-vision-preview", 
            "llama-3.2-90b-vision-preview", "gemma2-9b-it", "llama-3.1-8b-instant", "llama3-70b-8192", 
            "llama3-8b-8192", "deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-llama-70b"
        ], 
        index=0
    )

    if st.button("Select model for chatting"):
        st.session_state.selected_model = model_id
        st.success(f"Selected model: {model_id}")

    if st.session_state.selected_model:
        st.info(f"Current model: {st.session_state.selected_model}")

    # PDF Upload Section
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", 
        accept_multiple_files=True, 
        type=["pdf","docx", "txt"]
    )

    if not pdf_docs:
        st.warning('PLease insert PDF', icon="⚠️")
        st.stop()

    if st.button("Process", key="docs"):
        with st.spinner("Processing"):
            variable_list = ["context_list", "chat_history", "embedding_function", 
                    "vectorstore", "embedding_2d", "llm", "all_splits"]

            for variable in variable_list:
                st.session_state.setdefault(variable, None)
            
            history_upto_questions = 3
            st.session_state.context_list = LimitedSizeList(history_upto_questions)
            st.session_state.chat_history = LimitedSizeList(2*history_upto_questions)
            
            load_dotenv()
            # Access the API key
            # api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            st.session_state.embedding_function = get_embeddings()
            st.session_state.llm = get_llm(st.session_state.selected_model)
            # get text
            raw_text = get_texts_from_files(pdf_docs)
            # split text into chunks
            st.session_state.all_splits = split_text_into_chunks(raw_text)
            # create vectorstore
            st.session_state.vectorstore = get_vectorstore(st.session_state.all_splits,
                                                           st.session_state.embedding_function)
            # get docs embedding 
            st.session_state.embeddings_2d = False

            # selected_options = st.radio("calculate separate docs embeddings to visuallize query using PaCMAP", options)
            if st.session_state.embeddings_2d:
                st.session_state.embeddings_2d = get_all_docs_embedding(st.session_state.embedding_function,
                                                                        st.session_state.all_splits)
            st.switch_page("pages/chat.py")

if __name__ == "__main__":
    main()
