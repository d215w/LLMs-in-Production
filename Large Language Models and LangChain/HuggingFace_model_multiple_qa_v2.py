from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain
from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# intitialize Hub LLM
hub_llm = HuggingFaceHub(
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature': 0}
)

multi_template = """Answer the following questions one at a time

Question:
{questions}

Answers:
"""

long_prompt = PromptTemplate(template=multi_template, input_variables=['questions'])

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=long_prompt,
    llm=hub_llm
)

# user questions
qa_str = (
    "What color is a ripe banana?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
    "What is the capital of France\n"
)

# ask the uester a question
res = llm_chain.run(qa_str)
print(res)

# See terminal for results in list format