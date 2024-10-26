from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage,HumanMessage,SystemMessage

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

# use a list to store messages

chat_history =[]


# set an initial system message like the role of the model here (eg you are a professional in xyz)

system_message = SystemMessage(content="You are a helpful AI assistant")
chat_history.append(system_message)

# Chat Loop

while True:
    query = input("You:")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) # add user message

    # Get Ai response using history
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content)) # add ai message

    print(f"AI: {response.content}")

print("---- Message History ----")
print(chat_history)


