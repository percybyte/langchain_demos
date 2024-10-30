import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate # Para modelos ChatModel
# from langchain_core.prompts import PromptTemplate - Para modelos LLM

from langchain_core.prompts import FewShotChatMessagePromptTemplate # Para usar con Few Shot

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# Instancia del modelo
# Podemos conectarnos a un modelo LLM o un ChatModel
chatModel = ChatOpenAI(model="gpt-4o-mini")


# Creación del Prompt Template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un experto en {tema}"),
        ("human", "Hola, puedo hacerle una pregunta?"),
        ("ai", "Por supuesto?"),
        ("human", "{entrada_usuario}"),
    ]
)


# Ejecutar prompt usando la plantilla
messages = prompt_template.format_messages(
    tema="Futbol",
    entrada_usuario="5 Balones de Oro"
)


response = chatModel.invoke(messages)

print(response)


