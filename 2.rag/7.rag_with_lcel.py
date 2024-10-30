import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data/state_of_the_union.txt")


loaded_document = TextLoader(file_path).load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks_of_text = text_splitter.split_documents(loaded_document)
embeddings_model = OpenAIEmbeddings()
vector_db = FAISS.from_documents(chunks_of_text, embeddings_model)

retriever = vector_db.as_retriever(search_kwargs={"k": 1})

template = """Responde la pregunta basada solo en el siguiente contexto:

{context}

Pregunta: {pregunta}

"""

prompt = ChatPromptTemplate.from_template(template)

chatModel = ChatOpenAI(model="gpt-4o-mini")


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "pregunta": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser
)

response = chain.invoke("what did he say about ketanji brown jackson?")
