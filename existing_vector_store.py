from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

load_dotenv()  # take environment variables from .env.
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_API_KEY")

# instantiate the LLM and embedding models
llm = OpenAI(model='text-davinci-003', temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

