from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool

load_dotenv()


def current_time(*args, **kwargs):
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M:%P")


def search_wikipedia(query):
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return "I cound not find the wikipedia page"


tools = [
    Tool(
        name="Time",
        func=current_time,
        description="useful for when you need to know the current time",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="useful for when you need to know information about a topic",
    )
]

prompt = hub.pull("hwchase17/structured-chat-agent")

llm = ChatOpenAI(model='gpt-3.5-turbo')

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
