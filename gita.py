import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

def load_gita_text(file_path="gita.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_gita_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embedding_model)
    vectorstore.save_local("gita_faiss_db")
    return vectorstore

def ensure_vector_store():
    if not os.path.exists("gita_faiss_db"):
        print("Creating FAISS DB...")
        chunks = chunk_gita_text(load_gita_text())
        create_vector_store(chunks)

def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("gita_faiss_db", embedding_model, allow_dangerous_deserialization=True)

def get_qa_chain():
    ensure_vector_store()
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        streaming=True
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
