__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import wget

# Initialize the Llama model
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 30},
    model_kwargs={"load_in_8bit": True}
)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists("/root/.cache/warandpeace.txt"):
    wget.download("https://archive.org/download/warandpeace030164mbp/warandpeace030164mbp_djvu.txt", "/root/.cache/warandpeace.txt")
loader = TextLoader("/root/.cache/warandpeace.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create and persist the vector store
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="/root/.cache/chroma_db")
vectorstore.persist()

# Create a retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Function to handle user queries
def process_query(query):
    result = qa_chain({"query": query})
    return result["result"]

# Example usage
user_query = "Ignat is caught grinning in front of a mirror by Mavra Kuzminichna, and then he has to make tea for which of his relatives?"

print("Initial query:", user_query)

while True:
    if user_query.lower() == 'quit':
        break
    response = process_query(user_query)
    print("Answer:", response)
    user_query = input("Enter your query: ")
