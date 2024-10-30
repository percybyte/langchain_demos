"""
Retriever: Componente que busca y recupera información de una fuente de datos
como una base de datos vectorial. Se usa para encontrar documentos similares
o relevantes en función a una consulta, en donde generalmente convierte tanto
la consulta como las fuentes de datos en vectores para encontrar similitudes.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
chatModel = ChatOpenAI(model="gpt-4o-mini")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data/state_of_the_union.txt")


loaded_document = TextLoader(file_path).load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks_of_text = text_splitter.split_documents(loaded_document)
embeddings_model = OpenAIEmbeddings()
vector_db = FAISS.from_documents(chunks_of_text, embeddings_model)

# Usamos el retriever
# El parametro k=1 le indicara al retriever que solo devuelva el resultado
# mas relevante o similar
retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# Tarea: Que diferencia hay entre usar 'as_retriever' o 'similarity_search'


# Como lo podemos usar ahora con un LLM.
response = retriever.invoke("what did he say about ketanji brown jackson?")
