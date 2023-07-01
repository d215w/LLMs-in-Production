from langchain.chat_models import ChatOpenAI
from langchain.schema import (HumanMessage, SystemMessage)

from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

batch_messages = [
 [
    SystemMessage(content='You are a helpful assistant that translates English to French'),
    HumanMessage(content="Translate the following sentence: I love programming.")
 ],
 [
    SystemMessage(content='You are a helpful assistant that translates French to English'),
    HumanMessage(content="Translate the following sentence: J'aime la programmation.")
 ],
]

print(chat.generate(batch_messages))