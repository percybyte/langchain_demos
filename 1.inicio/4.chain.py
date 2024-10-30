import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# Inicializa el modelo de chat de OpenAI con el modelo especificado
chatModel = ChatOpenAI(model="gpt-4o-mini")

# Define ejemplos para el Few-Shot Learning, donde se muestra cómo se debería traducir del inglés al español
examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

# Crea el Prompt Template, definiendo la estructura del mensaje de ejemplo con variables input y output
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Encadena el Prompt Template con los ejemplos de Few-Shot Learning para guiar el modelo en la tarea
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Crea el Prompt final incluyendo un mensaje del sistema, el Few-Shot Prompt y un mensaje de entrada del usuario
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# Construye la cadena conectando el Prompt con el modelo de chat
chain = final_prompt | chatModel

# Ejecuta la cadena invocando la traducción con el input proporcionado
chain.invoke({"input": "Who was JFK?"})
