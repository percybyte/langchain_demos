"""
Uso de embeddings para transformar pequeñas partes de texto en números (vectores) para poder
guardarlos en una base de datos vectorial
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
chatModel = ChatOpenAI(model="gpt-4o-mini")

embeddings_model = OpenAIEmbeddings()

chunks_of_text = [
    "Hi there!",
    "Hello",
    "Whats your name?",
    "Bond, James Bond",
    "Hello, James Bond",
]


# Crear embedding de un documento (Simulación)
embeddings = embeddings_model.embed_documents(chunks_of_text)


# Crear embedding para la entrada del usuario
embedding_query = embeddings_model.embed_query("¿Cual es el nombre de la persona?")

# print(embeddings[0][:5])
print(len(embedding_query))

# NOTA: La fuente de información la hemos convertido en embeddings, así como la pregunta de un usuario.Pero como vemos la respuesta?
