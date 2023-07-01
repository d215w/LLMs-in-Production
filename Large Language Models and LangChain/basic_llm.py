from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003", temperature=0)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a kids company that makes {product}"
)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("golf clubs"))