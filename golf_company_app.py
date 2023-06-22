from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model='text-davinci-003', temperature=0.9)
prompt = PromptTemplate(
    input_variables=['product'],
    template='What is a good name for a golf company that makes {product}?'
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain by only specifying the input variable
print(chain.run('eco-friendly golf apparel'))