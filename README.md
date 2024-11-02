## Conversational 

These are core message types in LangChain for structuring chat interactions:

1. HumanMessage
- Represents user input
- Like: "What's the weather?"
- Used in chat prompts
```python
HumanMessage(content="Hello AI")
```

2. AIMessage  
- Represents AI assistant responses
- Previous outputs in conversation
```python
AIMessage(content="Hello human, how can I help?")
```

3. SystemMessage
- Sets behavior/context for AI
- Like personality, rules, or context,role
- Usually used at start of chat
```python 
SystemMessage(content="You are a helpful assistant")
```

Together they create chat history/memory:
```python
messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hi"),
    AIMessage(content="Hello!")
]
```

These are the building blocks for chat-based interactions in LangChain.


### Different ChatModels
```
Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/
```

## Prompt Templates

**Prompt Template Docs:**
```
https://python.langchain.com/v0.2/docs/concepts/#prompt-templateshttps://python.langchain.com/v0.2/docs/concepts/#prompt-templates
```
## Chains in Langchain

Chains in LangChain combine multiple components to create a sequence of operations. Here are the key types:

1. LLMChain
- Most basic chain
- Combines prompt template + LLM
```python
chain = LLMChain(llm=llm, prompt=prompt)
chain.run(input)
```

2. SimpleSequentialChain
- Single input -> passes through steps -> single output
- Output of each step is input to next
```python
chain = SimpleSequentialChain(chains=[chain1, chain2])
```

3. SequentialChain  
- Multiple inputs/outputs
- More flexible than SimpleSequential
```python
chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=[...],
    output_variables=[...]
)
```

4. RouterChain
- Routes inputs to different chains based on conditions
- Like a switch statement for chains

5. TransformChain
- Custom transformations on data
- No LLM calls, just data processing

Common Pattern:
```python
prompt -> LLM -> Parser -> Next Step
```

Key benefit: Chains let you compose complex workflows from simple components.

### Pipe Operator 
The pipe operator ( | ) in LangChain lets you chain components together in a clean, readable way, similar to Unix pipes:

```python
# Instead of nested calls like:
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"input": "hello"})

# You can do:
prompt | llm 

# Complex chains:
prompt | llm | output_parser | next_chain

# With RunniableMap (parallel operations):
docs | RunniableMap(prompt | llm) | output_parser
```

Key uses:
1. Chain components
2. Transform data
3. Create pipelines
4. Parallel processing

Benefits:
- More readable code
- Less nesting
- Functional programming style
- Easy to modify pipeline steps

It's similar to pandas' pipe operator but for LLM operations.

### Runnable Lambdas

Runnable Lambdas refer to the ability to create and execute small, self-contained functions or code snippets in LangChain. These Runnable Lambdas are similar to the concept of "lambda functions" in programming, which are anonymous, inline functions that can be defined and executed on the fly.

In the context of LangChain, Runnable Lambdas provide a way to easily incorporate custom logic or functionality into your LangChain workflows, without the need to create a separate, standalone Python module or function.

Some key characteristics and use cases of Runnable Lambdas in LangChain include:

1. **Inline Functionality**: Runnable Lambdas allow you to define small, focused pieces of logic directly within your LangChain code, rather than having to define them in a separate Python file.

2. **Dynamic Customization**: You can use Runnable Lambdas to dynamically customize the behavior of your LangChain components or chains, based on specific requirements or inputs.

3. **Rapid Prototyping**: Runnable Lambdas make it easier to experiment and iterate on your LangChain workflows, as you can quickly add, modify, or remove small pieces of functionality without having to restructure your entire codebase.

4. **Modularity and Composability**: Runnable Lambdas can be composed together with other LangChain components, allowing you to build more complex and modular workflows.

Runnable Lambdas provide a flexible and convenient way to incorporate custom logic into your LangChain workflows, helping you to build more powerful and tailored language models and applications.

# RAG



### **persistent_directory**


In ChromaDB, the persistent_directory is the location where the database stores its data on disk permanently, rather than just in memory.

Key points:
- Without it, data exists only in memory (lost when program ends)
- With it, data persists between sessions
- Stores embeddings, metadata, and index structures
- Can be reloaded and reused later

Simple explanation:
```python
# In-memory (temporary):
db = chromadb.Client()

# Persistent (saved to disk):
db = chromadb.PersistentClient(path="./my_chroma_db")
```

Think of it like:
- In-memory = temporary notebook
- Persistent = actual filing cabinet

This is crucial for:
- Saving embeddings for reuse
- Not having to recreate DB every run
- Production applications
- Backing up data

The path you specify becomes your "database folder" where ChromaDB saves everything.

# Understanding and Implementing LangChain Agents: A Comprehensive Guide

## Introduction to Agents

Agents in LangChain are autonomous systems that combine language models with tools to solve complex tasks. Think of an agent as an AI assistant that can use various tools and make decisions about how to accomplish a given objective. Unlike simple chains or models, agents can determine which actions to take and in what order, making them powerful for tasks that require multiple steps or different approaches.

## Basic Components of an Agent

### Tools
Tools are the functions or capabilities that an agent can use. These can range from simple calculators to complex APIs. Here's a basic example of setting up tools:

```python
from langchain.agents import Tool
from langchain.utilities import DuckDuckGoSearchRun
from langchain.tools import WikipediaAPIWrapper

# Initialize tool instances
search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()

# Create tool definitions
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching the internet for current information"
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for getting detailed information from Wikipedia"
    )
]
```

### The Language Model
The language model serves as the brain of the agent, making decisions about which tools to use and how to interpret their outputs. Here's how to set it up:

```python
from langchain.llms import OpenAI

# Initialize with temperature=0 for more consistent outputs
llm = OpenAI(temperature=0)
```

## Creating Your First Agent

Let's start with a simple agent that can search the internet and use Wikipedia:

```python
from langchain.agents import initialize_agent, AgentType

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
question = "What is quantum computing and who are its pioneers?"
response = agent.run(question)
```

When you run this code, the agent will:
1. Analyze the question
2. Decide whether to search the internet or Wikipedia
3. Process the information
4. Provide a comprehensive answer

## Advanced Agent Implementation

Let's create a more sophisticated agent that can handle complex tasks:

```python
from langchain.memory import ConversationBufferMemory
from langchain.tools import PythonREPLTool
from langchain.prompts import PromptTemplate

# Initialize additional tools
python_repl = PythonREPLTool()

# Create a custom tool
class DataAnalysisTool(BaseTool):
    name = "Data Analyzer"
    description = "Analyzes numerical data and provides statistical insights"
    
    def _run(self, data: str) -> str:
        # Convert string input to numerical data and analyze
        import pandas as pd
        import json
        
        try:
            data = json.loads(data)
            df = pd.DataFrame(data)
            analysis = {
                "mean": df.mean().to_dict(),
                "median": df.median().to_dict(),
                "std": df.std().to_dict()
            }
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

# Create a custom prompt template
custom_prompt = """You are a sophisticated AI assistant with expertise in data analysis and research.
You have access to the following tools:

{tools}

When approaching a task:
1. Carefully analyze the request
2. Break down complex problems into steps
3. Use appropriate tools for each step
4. Synthesize information into a coherent response

Use this format:
Question: the input question
Thought: your reasoning about what needs to be done
Action: the tool to use
Action Input: what you pass to the tool
Observation: the tool's output
... (repeat as needed)
Thought: final reasoning
Final Answer: your complete response

Begin!

Question: {input}
Thought: """

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the advanced agent
advanced_agent = initialize_agent(
    tools=[
        Tool(
            name="Python REPL",
            func=python_repl.run,
            description="Execute Python code for calculations"
        ),
        DataAnalysisTool(),
        *tools  # Include our previous tools
    ],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    agent_kwargs={'prefix': custom_prompt},
    verbose=True
)
```

## Real-World Application Examples

### Example 1: Research Assistant
This example shows how to use an agent for comprehensive research:

```python
def research_topic(topic: str, depth: str = "basic"):
    """
    Conducts research on a given topic with specified depth
    """
    research_prompt = f"""
    Conduct a {depth} research analysis on {topic}.
    If basic: Focus on key definitions and main concepts.
    If detailed: Include historical context, current developments, and future implications.
    """
    
    return advanced_agent.run(research_prompt)

# Usage example
research_result = research_topic("artificial photosynthesis", "detailed")
```

### Example 2: Data Analysis Pipeline
Here's an example of using an agent for data analysis:

```python
def analyze_dataset(data: str):
    """
    Analyzes a dataset and provides insights
    """
    analysis_steps = [
        "1. Examine the data structure",
        "2. Perform statistical analysis",
        "3. Identify patterns and trends",
        "4. Generate visualizations if needed",
        "5. Provide actionable insights"
    ]
    
    prompt = f"""
    Analyze the following dataset:
    {data}
    
    Follow these steps:
    {'\n'.join(analysis_steps)}
    
    Provide a comprehensive analysis with visualizations where appropriate.
    """
    
    return advanced_agent.run(prompt)

# Example usage
sample_data = {
    "sales": [100, 150, 200, 180, 220],
    "costs": [80, 90, 120, 100, 130],
    "customers": [20, 25, 30, 28, 35]
}

analysis_result = analyze_dataset(str(sample_data))
```

## Best Practices and Tips

1. **Tool Description Clarity**
   - Write clear, specific tool descriptions
   - Include example inputs/outputs in descriptions
   - Specify limitations and requirements

2. **Error Handling**
```python
def safe_agent_execution(agent, prompt, max_retries=3):
    """
    Safely execute agent tasks with error handling
    """
    for attempt in range(max_retries):
        try:
            return agent.run(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Failed after {max_retries} attempts. Error: {str(e)}"
            continue
```

3. **Memory Management**
   - Use appropriate memory types for your use case
   - Clear memory when switching contexts
   - Consider implementing memory persistence for long-running applications

## Advanced Configurations

### Custom Agent Creation
For specialized use cases, you can create custom agents:

```python
class SpecializedAgent(Agent):
    @property
    def observation_prefix(self) -> str:
        """Prefix to use for observations."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to use for the llm."""
        return "Assistant: "

    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n"
        return thoughts
```

## Conclusion

Agents in LangChain provide a powerful framework for creating autonomous AI systems that can use tools and make decisions. By combining different tools, custom prompts, and memory systems, you can create sophisticated agents capable of handling complex tasks across various domains.

Remember to:
- Choose appropriate tools for your use case
- Design clear and specific prompts
- Implement proper error handling
- Consider memory requirements
- Test thoroughly with various inputs

