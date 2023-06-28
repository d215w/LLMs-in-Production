from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# define template for translation
translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["text"],
                                      template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

source_language = "English"
target_language = "French"

text = "LangChain provides many modules that can be used to build language model applications."


translated_text = translation_chain.predict(source_language=source_language,
                                           target_language=target_language,
                                           text=text)
print(translated_text)