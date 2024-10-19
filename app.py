import os
from dotenv import load_dotenv
load_dotenv()

import openai 
openai.api_key = os.getenv("OPEN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="Hi my name is yash and i am a cheif AI engineer")])

from langchain_core.messages import AIMessage
model.invoke(
[
    HumanMessage(content="HI my name is yashashree and i am a cheif AI engineer"),
    AIMessage(content="Hello Yash! It's great to meet you.  \n\nThat's an impressive title. As a large language model, I'm fascinated by the work AI engineers do. What are some of the most exciting projects you're working on? \n\nI'm always eager to learn more about the world of AI"),
    HumanMessage(content="Hey whats my name and what do i do?")
    
]

)

## message history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store ={}
#function to retrive the chat history
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model,get_session_history)
#whenever session id is given we will get ebtire chat message history base is an abstract class to store the message hoistory

config = {"configurable":{"session_id":"chat1"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi my name is Radha and i am a chief engineer")],
    config=config
)
response.content

with_message_history.invoke(
    [HumanMessage(content="what is my name?")],
    config=config
)

config1 = {"configurable":{"session_id":"chat2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi my name is Radha and i live in vrindavan")],
    config=config1
)
response = with_message_history.invoke(
    [HumanMessage(content= "What is my name")],
    config=config1
)
response.content

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Answer all questions to your best abaility"),
        MessagesPlaceholder(variable_name="messages"),

    ]
)

chain = prompt|model

chain.invoke({"messages":[HumanMessage(content="Hi my name is Yashashree")]})

with_message_history=RunnableWithMessageHistory(chain,get_session_history)

config2 = {"configurable":{"session_id":"chat3"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi my name is methi")],
    config=config2
)

## Add more complexity

prompt = ChatPromptTemplate.from_messages([
    ("system",
    "you are a helpful assistant. answer all the question to the best of your knowledge i n {language}"),
    MessagesPlaceholder(variable_name="messages"),

]

    
)

chain = prompt|model

response=chain.invoke({"messages":[HumanMessage(content="Hi my name is lata")],"language":"Hindi"})
response.content

with_message_history=RunnableWithMessageHistory(chain,
                                                get_session_history,
                                                input_messages_key="messages")

config3 = {"configurable":{"session_id":"chat4"}}
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="Hi, i am yashashree")],"language":"Hindi"},
    config=config3
)



response = with_message_history.invoke(
    {"messages": [HumanMessage(content="what is my name?")],"language":"Hindi"},
    config=config3
)         

print(response.content)
