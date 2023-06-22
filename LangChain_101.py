from langchain.llms import OpenAI
from dotenv import load_dotenv

# Best to use a temperature of 0.70 to 0.90
# Balance of creativity and reliability

llm = OpenAI(openai_api_key=OPENAI_API_KEY, model="text-davinci-003", temperature=0.9)