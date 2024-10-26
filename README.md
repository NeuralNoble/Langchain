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