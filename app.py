from fastapi import FastAPI, Request
import os
import openai
import langchain
import tempfile
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["https://agape-delta.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv(find_dotenv())
openai_api_key = os.getenv("API_KEY")

text_file_path = "1 John.txt"  # Path to your Bible verses text file

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self):
        self.documents = []

    def on_retriever_start(self, query: str, **kwargs):
        self.query = query

    def on_retriever_end(self, documents, **kwargs):
        self.documents = documents

def configure_qa_chain(text_file_path):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, os.path.basename(text_file_path))
    
    with open(temp_filepath, 'w', encoding='utf-8') as temp_file:
        with open(text_file_path, 'r', encoding='utf-8') as original_file:
            temp_file.write(original_file.read())
    
    loader = TextLoader(temp_filepath)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 2, "fetch_k": 4})
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613", temperature=0, openai_api_key=openai_api_key, streaming=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True 
    )
    return qa_chain

qa_chain = configure_qa_chain(text_file_path)

def generate_prompt(section, subtheme):
    prompts = {
        "meditations": f"Generate a meditation about {subtheme} with the following structure: \n\nTitle: \n\nBody: \n\nFurther Reading: Include the Bible verse where this information was retrieved from. Do not explicitly show the headings for 'title' and 'body'.",
        "narrative": f"Generate a full narrative about {subtheme}. The narrative should be engaging and cover the theme comprehensively. For further reading, include the Bible verse where this information was retrieved from. Do not explicitly show the headings for 'title' and 'body'.",
        "devotional": f"Generate a full devotional about {subtheme}. The devotional should include a title, an introductory section, a scripture passage, an explanation and reflection, personal application, practical steps, encouragement and assurance, and a conclusion. For further reading, include the Bible verse where this information was retrieved from. Do not explicitly write the word 'title'. All the other subheadings should be in bold."
    }
    if section.lower() not in prompts:
        raise ValueError("Invalid section. Please provide a valid section: 'meditations', 'narrative', or 'devotional'.")
    return prompts[section.lower()]

@app.get('/')
def read_root():
    return {"message": "Hello world"}

@app.post('/chat')
async def chat(request: Request):
    data = await request.json()
    user_query = data.get("query")
    retrieval_handler = PrintRetrievalHandler()
    stream_handler = StreamHandler()
    response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
    return {"response": stream_handler.text}

@app.post('/meditations')
async def meditations_handler(request: Request):
    data = await request.json()
    subtheme = data.get("subtheme")
    prompt = generate_prompt("meditations", subtheme)
    retrieval_handler = PrintRetrievalHandler()
    stream_handler = StreamHandler()
    response = qa_chain.run(prompt, callbacks=[retrieval_handler, stream_handler])
    return {"response": stream_handler.text}

@app.post('/narrative')
async def narrative_handler(request: Request):
    data = await request.json()
    subtheme = data.get("subtheme")
    prompt = generate_prompt("narrative", subtheme)
    retrieval_handler = PrintRetrievalHandler()
    stream_handler = StreamHandler()
    response = qa_chain.run(prompt, callbacks=[retrieval_handler, stream_handler])
    return {"response": stream_handler.text}

@app.post('/devotional')
async def devotional_handler(request: Request):
    data = await request.json()
    subtheme = data.get("subtheme")
    prompt = generate_prompt("devotional", subtheme)
    retrieval_handler = PrintRetrievalHandler()
    stream_handler = StreamHandler()
    response = qa_chain.run(prompt, callbacks=[retrieval_handler, stream_handler])
    return {"response": stream_handler.text}
