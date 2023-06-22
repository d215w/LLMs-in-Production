from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Best to use a temperature of 0.70 to 0.90
# Balance of creativity and reliability

llm = OpenAI(openai_api_key=OPENAI_API_KEY, model="text-davinci-003", temperature=0.9)

# text = "Suggest a personalized workout routine for someone looking to get more distnace in \
#     in their golf game."
# print(llm(text))



