__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import wget, torch
from transformers import StoppingCriteria, StoppingCriteriaList

from transformers import StoppingCriteria

class StopOnLinebreak(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the generated tokens
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # Check if any of the stop strings are in the decoded text
        for stop_string in self.stop_strings:
            if stop_string in decoded_text:
                return True
        return False

import transformers
llama_tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
stop_strings = ["\n",]
stopping_criteria = StoppingCriteriaList([
    StopOnLinebreak(tokenizer=llama_tokenizer, stop_strings=stop_strings)
])

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "stopping_criteria": stopping_criteria
    },
    model_kwargs={"load_in_8bit": True}
)

llm.pipeline.tokenizer.pad_token = llm.pipeline.tokenizer.eos_token

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load and prepare the data (unchanged)
if not os.path.exists("/root/.cache/warandpeace.txt"):
    wget.download("https://archive.org/download/warandpeace030164mbp/warandpeace030164mbp_djvu.txt", "/root/.cache/warandpeace.txt")

if not os.path.exists("/root/.cache/chroma_db"):
    loader = TextLoader("/root/.cache/warandpeace.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="/root/.cache/chroma_db")
    vectorstore.persist()
else:
    vectorstore = Chroma(persist_directory="/root/.cache/chroma_db", embedding_function=embeddings)

replacement_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an assistant for question-answering tasks.<|eot_id|><|start_header_id|>user<|end_header_id|>

Try to use the following document excerpts to answer the question. If you can't provide a reasonable answer grounded in the document excerpts, just say that you don't know.
Use three sentences maximum and keep the answer concise:
## Question: {query}

## Documents: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Helpful Answer:"""

# Create a retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
        "prompt": PromptTemplate(
            template=replacement_template,
            input_variables=["query", "context"],
        ),
    },
)

# Create a retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Function to handle user queries with manual stop sequence handling
def process_query(query):
    result = qa_chain({"query": query})
    response = result["result"]
    return response

# Example usage
user_query = "Ignat is caught grinning in front of a mirror by Mavra Kuzminichna, and then he has to make tea for which of his relatives?"
print("Initial query:", user_query)

while True:
    if user_query.lower() == 'quit':
        break
    response = process_query(user_query)
    print("Answer:",response)
    user_query = input("Enter your query (or 'quit' to exit): ")
