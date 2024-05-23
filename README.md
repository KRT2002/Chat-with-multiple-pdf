# Chat-with-multiple-pdf

## Introduction
------------
"Welcome to the Chat with Multiple PDF App! This Python application allows users to engage in conversations with multiple PDF documents using natural language. Simply upload your PDFs and begin asking questions relevant to the content. The app utilizes a language model from the open-source Hugging Face model hub to provide accurate responses tailored to the loaded documents. Experience seamless interaction and efficient information retrieval with MultiPDF Chat."

## How It Works
------------

1. Loading PDFs: The app starts by reading multiple PDF documents and extracting their text content.

2. Chunking Text: Following this, the extracted text is divided into smaller, more manageable chunks that can be effectively processed.

3. Leveraging a Language Model: Our application taps into a language model from the open-source Hugging Face model hub. This model helps generate vector representations (embeddings) of the text chunks, enabling better analysis.

4. Finding Similarities: When you ask a question, the app compares it with the text chunks. It then identifies the most relevant chunks by considering not just the current question but also past chat history. It does this using a history-aware retriever.

5. Generating Responses: Finally, the selected text chunks are handed over to the language model. This model, sourced from the Hugging Face model hub, generates a response based on the content of the PDFs.

6. Contextual Understanding: Additionally, the app allows users to access context documents and review chat history, enhancing the overall conversational experience.

## Dependencies and Installation
----------------------------

To install the Chat with Multiple PDF App, please adhere to the following instructions:

1. Clone the repository to your local machine.

2. Install the necessary dependencies by executing the subsequent command:

 ```
 pip install -r requirements.txt
 ```
This ensures that all required packages are correctly installed.

3. Obtain an API key from the Hugging Face model hub and incorporate it into the .env file located in the project directory.
```commandline
HUGGINGFACE_API_KEY=your_secret_api_key
```
This step is crucial for accessing the language model from the Hugging Face model hub.

## Usage
-----

To utilize the Chat with Multiple PDF App, please proceed with the following steps:

1. Ensure that you have installed the required dependencies and added the Hugging Face API key to the .env file.

2. Run the app.py file using the Streamlit CLI. Execute the subsequent command:
'''
streamlit run app.py
'''
This command initiates the application, and it will open in your default web browser, showcasing the user interface.

3. Load multiple PDF documents into the app by following the provided instructions.

4. Engage with the chat interface by asking questions in natural language about the loaded PDFs.
