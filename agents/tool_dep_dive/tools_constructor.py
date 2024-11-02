from langchain import hub
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel,Field
from langchain_core.tools import Tool,StructuredTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def greet_user(name:str) -> str:
    return f"Hello {name}"

def reverse_string(string:str) -> str:
    return string[::-1]

def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b


class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")


tools = [
    # Use Tool for simpler functions with a single input parameter.
    # This is straightforward and doesn't require an input schema.
    Tool(
        name="GreetUser",  # Name of the tool
        func=greet_user,  # Function to execute
        description="Greets the user by name.",  # Description of the tool
    ),
    # Use Tool for another simple function with a single input parameter.
    Tool(
        name="ReverseString",  # Name of the tool
        func=reverse_string,  # Function to execute
        description="Reverses the given string.",  # Description of the tool
    ),
    # Use StructuredTool for more complex functions that require multiple input parameters.
    # StructuredTool allows us to define an input schema using Pydantic, ensuring proper validation and description.
    StructuredTool.from_function(
        func=concatenate_strings,  # Function to execute
        name="ConcatenateStrings",  # Name of the tool
        description="Concatenates two strings.",  # Description of the tool
        args_schema=ConcatenateStringsArgs,  # Schema defining the tool's input arguments
    ),
]

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=llm,  # Language model to use
    tools=tools,  # List of tools available to the agent
    prompt=prompt,  # Prompt template to guide the agent's responses
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,  # The agent to execute
    tools=tools,  # List of tools available to the agent
    verbose=True,  # Enable verbose logging
    handle_parsing_errors=True,  # Handle parsing errors gracefully
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)