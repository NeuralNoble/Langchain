from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (AgentExecutor, create_react_agent)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()


def get_current_time(*args, **kwargs):
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M:%P")


# List of tools available to the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="useful for when you need to know the current time",

    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent= agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({"input": "What time is it?"})

print(response)
