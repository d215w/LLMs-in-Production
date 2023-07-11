from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initiliaze LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n"""

example_prompt = PromptTemplate(
    input_variables=("color", "emotion"),
                     template=example_formatter_template,
    )

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotion associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")

# create the LLMChain for the prompt
chain = LLMChain(llm=llm,
                 prompt=PromptTemplate(template=formatted_prompt,
                                       input_variables=[]))

# Run the LLMChain to get the new content
response = chain.run({})

print("Color: purple")
print("Emotion:", response)