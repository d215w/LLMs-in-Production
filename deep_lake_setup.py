from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()  # take environment variables from .env.
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_API_KEY")

# instantiate the LLM and embedding models
llm = OpenAI(model='text-davinci-003', temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# create our documents
texts = [
    'Napoleon Bonapart was born 15 August 1769',
    'Louis XIV was born in 5 September 1638'
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
my_activeloop_org_id = 'nbeaudoin'
my_activeloop_dataset_name = 'langchain_course_zero_to_hero'
dataset_path =f'hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}'
db = DeepLake(dataset_path=dataset_path,
              embedding_function=embeddings)

# add documents to Deep Lake dataset
db.add_documents(docs)