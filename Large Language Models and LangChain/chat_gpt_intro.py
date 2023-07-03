from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage,
                              HumanMessage,
                              AIMessage)

from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)

# add to messages
messages.append(prompt)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

reponse = llm(messages)