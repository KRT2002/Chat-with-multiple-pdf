import streamlit as st
import multiprocessing
from PyPDF2 import PdfReader
import pacmap
import plotly.express as px
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import random
import string


# Function to process a chunk of pages
def process_chunk(chunk_start, chunk_end, pdf_file):
    chunk_text = ""
    for page_num in range(chunk_start, chunk_end):
        page = pdf_file.pages[page_num]
        page_text = page.extract_text()
        chunk_text += page_text
    return chunk_text

def get_texts_from_pdf(uploaded_files):

    text = ""

    for uploaded_file in uploaded_files:

        # Open the uploaded PDF file
        pdf_file = PdfReader(uploaded_file)

        # Process each page of the PDF file in chunks
        total_pages = len(pdf_file.pages)
        chunk_size = 10  # Number of pages to process in each chunk

        # Determine the number of CPU cores
        num_cores = multiprocessing.cpu_count()

        # Divide the pages into chunks for parallel processing
        chunks = [(start, min(start + chunk_size, total_pages)) for start in range(0, total_pages, chunk_size)]
        # Create a multiprocessing Pool
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Process the PDF file in parallel using multiprocessing Pool
            chunk_texts = pool.starmap(process_chunk, [(start, end, pdf_file) for start, end in chunks])
        
        # Concatenate the text from all chunks
        text += ''.join(chunk_texts)

    return text

def get_chunk_docs_embedding(embeddings, chunk_start, chunk_end, all_splits):
   chunk_list = []
   for idx in range(chunk_start, chunk_end):
      chunk_list.append(list(embeddings.embed_documents(all_splits[idx])))
   return chunk_list

def get_all_docs_embedding(embeddings, all_splits):

    total_docs = len(all_splits)
    chunk_size = 10  # Number of docs to process in each chunk
    # Determine the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Divide the pages into chunks for parallel processing
    chunks = [(start, min(start + chunk_size, total_docs)) for start in range(0, total_docs, chunk_size)]
    # Create a multiprocessing Pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Process the PDF file in parallel using multiprocessing Pool
        chunk_docs_embeddings = pool.starmap(get_chunk_docs_embedding, 
                                             [(embeddings, start, end, all_splits) for start, end in chunks])
    # merge list of chunks into single list
    docs_embeddings = list(chain.from_iterable(chunk_docs_embeddings))
    return docs_embeddings

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        add_start_index=True
    )
    all_splits = text_splitter.split_text(text)
    return all_splits

def pretty_print_docs(docs):
    return (
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def get_embeddings(HUGGINGFACEHUB_API_TOKEN):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key = HUGGINGFACEHUB_API_TOKEN,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

def get_vectorstore(all_splits, embeddings):
    """Generate a random string of specified length."""
    length = 10
    name = ''.join(random.choices(string.ascii_letters, k=length))
    vectorstore = Chroma.from_texts(texts=all_splits,
                                    embedding=embeddings,
                                    collection_name = name)
    return vectorstore
    
def get_contexual_retriever(vectorstore, embeddings):
    # cleanup the vector database
    # vectorstore.delete_collection()
    retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 6})
    
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever
    )

    return compression_retriever

def get_llm(HUGGINGFACEHUB_API_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens = 512,
        top_k = 50,
        temperature = 0.1,
        repetition_penalty = 1.03,
        huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN
    )
    return llm

def history_aware_retriever_chain(llm, compression_retriever):

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def qa_chain(llm):

    qa_system_prompt = """
    Please provide concise assistance for the user's query. \
    If context is not provided, explicitly state 'I don't know' instead of attempting to answer the question. \
    Limit your response to three sentences at most for brevity.

    <context>
    {context}
    </context>

    Question: {input}"""


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return question_answer_chain


# function which question answering chain only when context is provided
# return llm response along with retrieved documents, chat history
def question_answering_function(user_query, chat_history, history_aware_retriever, question_answer_chain):
  context_ = history_aware_retriever.invoke({
      "chat_history": chat_history,
      "input": user_query
  })

  if len(context_)!=0:
    result = (question_answer_chain.invoke({
        "input": user_query,
        "context": context_
    }))
    if "Assistant:" in result:
       result = result.split("Assistant:")[1].strip()
    chat_history.insert(AIMessage(content = result))
    chat_history.insert(HumanMessage(content = user_query))
    return result, context_, chat_history

  else:
    result = "I don't have information about your specific query. Please provide more details so I can assist you better. Without context, it's impossible to answer your question."
    return result, context_, chat_history
  

# search user query inside chat history if found return llm response else call question_answering_function (above one)
# return llm response, context (regarding user query), chat history, context list (regarding user query in chat history)
def previous_question(user_query, chat_history, context_list, embeddings, history_aware_retriever, question_answer_chain):

  if len(chat_history)!=0:
    embeddings_1 = np.array(embeddings.embed_query(user_query)).reshape(1,384)
    for i in range(0, len(chat_history), 2):
      embeddings_2 = np.array(embeddings.embed_query(chat_history[i].content)).reshape(1,384)
      similarity_matrix = cosine_similarity(embeddings_1, embeddings_2)
      if similarity_matrix[0][0] > 0.9:
        return (chat_history[i+1].content, context_list[int(i/2)], chat_history, context_list)

  result, context_, chat_history = question_answering_function(user_query, chat_history, history_aware_retriever, question_answer_chain)
  if len(context_)==0:
    return (result , context_, chat_history, context_list)

  context_docs = pretty_print_docs(context_)
  context_list.insert(context_docs)
  return (result , context_docs, chat_history, context_list)

class LimitedSizeList(list):
    def __init__(self, max_size):
        super().__init__()  # Initialize the list
        self.max_size = max_size  # Maximum allowed size of the list

    def insert(self, item):
        if len(self) >= self.max_size:
            self.pop()  # removes last item in list
        super().insert(0, item)  # Insert the new item at the beginning of the list

def answer_user_query(user_query, chat_history, context_list, history_aware_retriever, question_answer_chain, embeddings):
    
    result , context_docs, chat_history, context_list = previous_question(user_query, chat_history, context_list, embeddings, history_aware_retriever, question_answer_chain)
    if result != "I don't have information about your specific query. \
                  Please provide more details so I can assist you better. \
                  Without context, it's impossible to answer your question.":
        return result, context_docs, chat_history, context_list
    return result, context_docs, chat_history, context_list
    
def visuallize_query(embeddings, user_query, embeddings_2d, all_splits):
    # Initialize the PaCMAP object with desired parameters
    embedding_projector = pacmap.PaCMAP(n_components=2,
                                        n_neighbors=None,
                                        MN_ratio=0.5,
                                        FP_ratio=2.0,
                                        random_state=1)

    query_vector = embeddings.embed_query(user_query)
    embeddings_2d_q = embeddings_2d + [query_vector]

    # fit the data (The index of transformed data corresponds to the index of the original data)
    documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d_q), init="pca")

    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": "sensor 1.pdf",
                "extract": all_splits[i][:100] + "...",
                "symbol": "circle",
                "size_col": 4,
            }
            for i in range(len(all_splits))
        ]
        + [
            {
                "x": documents_projected[-1, 0],
                "y": documents_projected[-1, 1],
                "source": "user_query",
                "extract": user_query,
                "size_col": 100,
                "symbol": "star",
            }
        ]
    )

    # visualize the embedding
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={"User query": "black"},
        width=1000,
        height=700,
    )

    fig.update_traces(
        marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        legend_title_text="<b>Chunk source</b>",
        title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
    )

    # fig.show()
    st.plotly_chart(fig, use_container_width = True)
