from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# load environment variables from .env

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

# invoke the model with a message

response = model.invoke("what is your name ?")
print("Full Result:")
print(response)
print("content only:")
print(response.content)