from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain
from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

template = """Question: {question}

Answer:"""
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "What is the capital of France?"

# intitialize Hub LLM
hub_llm = HuggingFaceHub(
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature': 0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask teh user question about the capital of France
print(llm_chain.run(question))