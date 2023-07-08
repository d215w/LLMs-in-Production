from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003",
             temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""

prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template
)

# Input data for the prompt
input_data = {"theme": "interstellar travel",
              "year": "3030"}

# create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# run the LLMChain to get the AI-generated song title
response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)