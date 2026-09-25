from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

client = Client()

promptName = "sum"

prompt = client.pull_prompt(promptName)
print("=======================")
print(prompt)
print("=======================")
