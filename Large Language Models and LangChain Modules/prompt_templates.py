from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    },
    {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt examples from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# break prompt into prefix and suffix
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing 
entertaining and amusing responses to users' questions. Here are some
examples:
"""

# the suffix 
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Time to run it!

# Load the model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
print(chain.run("What's the meaning of life"))

