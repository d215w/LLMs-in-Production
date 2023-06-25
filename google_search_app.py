from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

# load credentials
load_dotenv()  # take environment variables from .env.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

llm = OpenAI(model='text-davinci-003', temperature=0)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name='google-search',
        func=search.run,
        description='useful for when you need to search google to answer questions about current events'
    )
]

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         max_iterations=6)

response = agent("What's the latest news on the Titan sub?")
print(response["output"])