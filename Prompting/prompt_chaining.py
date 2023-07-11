from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initialize the LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Prompt 1
template_question = """What is the name of the famous scientist who discovered the theory of general relativity?
Answer:"""
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of relativity
Answer:"""
prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt on an empty dictionary
response_question = chain_question.run({})

# create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# extract the scientist's name from the response
scientist = response_question.strip()

# input data for the second prompt
input_data = {"scientist:", scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)
