from dotenv import load_dotenv
from langchain_core.messages import AIMessage , HumanMessage , SystemMessage
from langchain_openai  import ChatOpenAI

load_dotenv()

# model selection and loading
model = ChatOpenAI(model="gpt-3.5-turbo")



# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
# HumanMessage:
#   Message from a human to the AI model.

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
]

# Invoke the model with message

response = model.invoke(messages)
print(f"answer from AI: {response.content}")


