from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages(
   [
        ("system", "you are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {jokes_count} jokes."),
   ]
)


uppercase = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase | count_words

result = chain.invoke({"topic":"dogs","jokes_count":5})

print(result)