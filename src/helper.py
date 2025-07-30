import re
import os
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# PDF Loader
def load_pdf_docs(data_dir):
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

# Cleaning
def clean_text(text):
    text = re.sub(r"GEM\s*-\s*\d+\s*to\s*\d+\s*-\s*\w.*?Page\s*\d+.*?GALE ENCYCLOPEDIA OF MEDICINE.*?\w*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"GALE ENCYCLOPEDIA OF MEDICINE", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s+\d+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# Chunking
def chunk_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Main RAG chain builder
def build_rag_chain(pdf_folder_path):
    extracted_docs = load_pdf_docs(pdf_folder_path)
    
    cleaned_docs = [
        Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
        for doc in extracted_docs
    ]
    
    text_chunks = chunk_text(cleaned_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("medicalbot")

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_length=512,
    device=-1  # Force CPU usage, avoids meta tensor issue
)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)


    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )

    return rag_chain

from dotenv import load_dotenv
import os

# Force .env load from root directory (one level up from /src/)
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env')))


