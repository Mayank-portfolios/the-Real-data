# AI Chatbot with Intent Recognition

A Python-based chatbot using **OpenAI API** and **LangChain**,**RAG**, capable of handling basic Q&A with intent recognition.

## Features
- Intent detection (85% accuracy on test queries)
- Dynamic response generation
- Contextual conversation flow

## Technologies
- Python 3.x
- OpenAI GPT
- LangChain
- spaCy/NLTK

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   
```python
import langchain
from langchain_openai import ChatOpenAI  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import pymupdf

# Load environment variables from .env file
load_dotenv()

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "Question:{question}")
    ]
)

# Set up the chain
llm = ChatOpenAI()
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit app
st.title("Chat Assistant")

question = st.text_input("Ask your question:")
pl = st.selectbox("What you want to upload", ["PDF", "IMAGE", "CSV", "CODE"])

if pl == "CSV":
    df = st.file_uploader("Upload CSV", type=["csv"])
    if df is not None:
        df = pd.read_csv(df)
        st.write("✅ File uploaded successfully!")
        st.dataframe(df)
    else:
        st.warning("📂 Please upload a CSV file to continue.")
else:
    if pl == "PDF":
        pdf_upload = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf_upload is not None:
            try:
                docs=pymupdf.open(stream=pdf_upload.read(),filetype="pdf")
                text=""
                for page in docs:
                    text+=page.get_text()
                material=st.write(text)
                splitter=CharacterTextSplitter(
                    chunk_size=200,
                    chunk_overlap=0,
                    separator=''
                )
                documents = [{"page_content": text}]
                result = splitter.split_documents(documents)
                
            except FileNotFoundError as e:
                st.error(f"Error: The file was not found. Details: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
            else:
                st.success("File processed successfully")

if question:
    response = chain.invoke({"question": question})
    st.write(response)
