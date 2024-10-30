import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
chatModel = ChatOpenAI(model="gpt-4o-mini")
embeddings_model = OpenAIEmbeddings()

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data/be-good.txt")


loaded_document = TextLoader(file_path).load()

# Creamos un splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Divimos el documento en pequeñas porciones con un splitter
chunks_of_text = text_splitter.split_documents(loaded_document)

# Creamos una base de datos vectorial para almacenar los datos embebidos
vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())


# Interacción del usuario
question_user = "¿Cual es el título del documento?"

response = vector_db.similarity_search(question_user[0])

print(response)
