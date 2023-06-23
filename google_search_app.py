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

llm = OpenAI(model='text-davinci-003', temperature=0)

search = GoogleSearchAPIWrapper()

