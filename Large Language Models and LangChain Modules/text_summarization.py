from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# define template for summarization
summarization_template = "Summarize the following text to one sentence: {text}"
summarization_prompt = PromptTemplate(input_variables=["text"],
                                      template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

# call predict function 
text = "LangChain provides many modules that can be used to build language model applications. \
        Modules can be combined to create more complex applications, or be used individuallty for simple \
        applications. The most basic building block of LangChian is calling an LLM on some input. \
        Let's walk through a simple example of how to do this. For this purpose, let's pretend we are building \
        a sercice that generates a company name based on what the company makes."

summarized_text = summarization_chain.predict(text=text)
print(summarized_text)