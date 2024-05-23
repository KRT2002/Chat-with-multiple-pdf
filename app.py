import streamlit as st
from dotenv import load_dotenv
import os

from functions import (LimitedSizeList,
                       get_texts_from_pdf,
                       split_text_into_chunks,
                       get_vectorstore,
                       get_embeddings,
                       get_llm, 
                       get_all_docs_embedding)

def main():

    st.set_page_config("Chat PDF")

    st.header("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type="pdf")

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
            api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            st.session_state.embedding_function = get_embeddings(api_key)
            st.session_state.llm = get_llm(api_key)
            # get text
            raw_text = get_texts_from_pdf(pdf_docs)
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
