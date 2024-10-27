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