from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {jokes_count} jokes."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic":"lawyers","jokes_count":3})

print(result)


