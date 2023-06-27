from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

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
prefix = """The following are excerpts"""