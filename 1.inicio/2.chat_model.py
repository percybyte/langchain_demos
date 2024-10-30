import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI


_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# Instancia del modelo
chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


messages = [
    ("system","Eres un experto en desarrollo de software"),
    ("human", "Dime un lenguaje de programación muy antiguo")
]


response = chatModel.invoke(messages)


print(response)

