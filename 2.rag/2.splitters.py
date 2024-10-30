import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredHTMLLoader, PyPDFLoader, WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# Usando un Chat Model
chatModel = ChatOpenAI(model="gpt-4o-mini")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data/be-good.txt")


loader = TextLoader(file_path)
loaded_data= loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1024,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

texts = text_splitter.create_documents([loaded_data[0].page_content])

print(len(texts))