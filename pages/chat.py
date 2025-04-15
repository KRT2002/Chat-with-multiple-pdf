import streamlit as st
import time
from functions import (get_contexual_retriever,
                       answer_user_query,
                       history_aware_retriever_chain,
                       qa_chain,
                       visuallize_query,
                       get_llm)

def stream_data():
    for word in result.split(" "):
        yield word + " "
        time.sleep(0.05)

st.set_page_config("QA with PDF", page_icon="🧊")

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

    st.header("Chat with PDF 💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("answer"):
        # Initialize the model
        st.session_state.llm = get_llm(st.session_state.selected_model)

        # create contextual retriever
        contextual_retriever = get_contexual_retriever(st.session_state.vectorstore, 
                                                    st.session_state.embedding_function)
        # create history aware retriever
        history_aware_retriever = history_aware_retriever_chain(st.session_state.llm, 
                                                                contextual_retriever)
        # create question answering chain
        question_answer_chain = qa_chain(st.session_state.llm)
        # st.write(question_answer_chain.invoke({"input":"what is sensor?","context":[]}))
        result, context_docs, chat_history, context_list = answer_user_query(user_question, 
                                                                            st.session_state.chat_history, 
                                                                            st.session_state.context_list, 
                                                                            history_aware_retriever, 
                                                                            question_answer_chain, 
                                                                            st.session_state.embedding_function) 

        # tab creation
        if st.session_state.embeddings_2d :
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["QA", "context docs", 
                                                    "chat history", "context documents", 
                                                    "query visuallization"])

            with tab5:
                # visuallize query
                with st.expander("visuallize query embeddings"):
                    visuallize_query(st.session_state.embedding_function, 
                                    user_question, 
                                    st.session_state.embeddings_2d, 
                                    st.session_state.all_splits)

        else:
            tab1, tab2, tab3, tab4 = st.tabs(["QA", "context documents", 
                                            "chat history", "context list"])

        with tab1:
            st.write_stream(stream_data)
            st.toast("Certainly! Your query has been answered. Is there anything else you'd like to know?", icon='😍')
            
        with tab2:
            with st.expander("context documents"):
                st.text(context_docs)
        
        with tab3:
            with st.expander("chat history"):
                st.write(chat_history)
        
        with tab4:
            with st.expander("context list"):
                st.write(context_list)

        


