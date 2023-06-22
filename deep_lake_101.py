from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_API_KEY")
