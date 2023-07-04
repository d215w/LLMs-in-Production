from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# Load the OpenAI key
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model='text-davinci-003', temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# start the conversation
conversation.predict(input='Tell me about yourself')
conversation.predict(input='How can you help me with my golf swing?')

# Display the convo
print(conversation)