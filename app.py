import PyPDF2
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Create sidebar
with st.sidebar:
  st.title('📄💬 PDF ChatBot')
  st.markdown("""
  ## About
  This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
  
  """)
  add_vertical_space(5)
  st.write('Made by [Azamat](https://github.com/Azamat0315277)')

def main():
  st.header('Chat with PDF 💬')
  load_dotenv()

  # upload a PDF file
  pdf = st.file_uploader('Upload your PDF', type='pdf')
  st.write(pdf.name)

  if pdf is not None:
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()


    # Split text 
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
    )

    chunks = text_splitter.split_text(text=text)

    # Embeddings

    
    store_name = pdf.name[:-4]

    # Check if file exists
    if os.path.exists(f"chatpdf/{store_name}.pkl"):
        with open(f"chatpdf/{store_name}.pkl", 'rb') as f:
            vectorstore = pickle.load(f)
        #st.write('Embeddings Loaded from the disk')
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"chatpdf/{store_name}.pkl", 'wb') as f:
            pickle.dump(vectorstore, f)
        #st.write('Embeddings Computing Comleted')

    # Accept user question/query
    query = st.text_input('Ask questions about your PDF file:')
    #st.write(query)

    if query:
        docs = vectorstore.similarity_search(query=query, k=3)
        llm=ChatOpenAI(temperature=0)
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        #check pricing of response by `get_openai_callback`
        with get_openai_callback() as cb:
           
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

    


if __name__ == '__main__':
  main()