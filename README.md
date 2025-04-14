# AI-Powered Document Interaction Application (v2)

## Introduction
------------
Welcome to the **AI-Powered Document Interaction Application – Version 2**! This advanced Python application allows users to interact with multiple documents using natural language. Whether it is a PDF, Word document, or plain text file, you can upload and query your documents with ease. Version 2 introduces enhanced flexibility by supporting multiple file formats and allowing users to select their preferred language model from a range of available model (gemini, deepseek, open source llm, etc.). Enjoy an intuitive, customizable, and efficient information retrieval experience with MultiDoc Chat.

## How It Works
------------

1. **Uploading Documents**: Users can upload documents in PDF, DOC/DOCX, or TXT formats. The app automatically detects the format and extracts readable text content from each file.

2. **Chunking Text**: The extracted text is segmented into manageable chunks to ensure accurate and efficient processing by the language model.

3. **Multi-Model Language Support**: Users can now choose from multiple language models integrated from the Hugging Face model hub. This empowers users to tailor the experience to their performance or accuracy preferences.

4. **Creating Embeddings**: The model  generates vector embeddings for all text chunks, facilitating semantic understanding.

5. **Intelligent Retrieval**: When a user asks a question, the app uses a history-aware retriever to find the most relevant text chunks by considering both the current query and the prior conversation context.

6. **Generating Responses**: The retrieved chunks are passed to the chosen language model to generate informative, context-aware responses based on the content of the uploaded documents.

7. **Contextual Review**: Users can view supporting context documents and revisit previous conversations for a seamless and informed dialogue.


## Dependencies and Installation
----------------------------

To install the AI-Powered Document Interaction Application – Version 2, please adhere to the following instructions:

1. Clone the repository to your local machine. 
 ```
 git clone https://github.com/KRT2002/Chat-with-multiple-pdf.git

 cd Chat-with-multiple-pdf

 git checkout main-v2
 ```

2. Install the necessary dependencies by executing the subsequent command:

 ```
 pip install -r requirements.txt
 ```
   This ensures that all required packages are correctly installed.

3. Obtain an API key from the Hugging Face model hub and incorporate it into the .env file located in the project directory.

```commandline
HUGGINGFACE_API_KEY=your_secret_api_key
GOOGLE_API_KEY=your_secret_api_key
GROQ_API_KEY=your_secret_api_key
```
   This step is crucial for accessing the language model from the gemini and groqcloud.

## Usage
-----

To utilize the AI-Powered Document Interaction Application – Version 2, please proceed with the following steps:

1. Ensure that you have installed the required dependencies and added the required API key to the .env file.

2. Run the app.py file using the Streamlit CLI. Execute the subsequent command:

 ```
streamlit run app.py
 ```
This command initiates the application, and it will open in your default web browser, showcasing the user interface.

3. Load multiple documents into the app by following the provided instructions.

4. Engage with the chat interface by asking questions in natural language about the loaded PDFs.
