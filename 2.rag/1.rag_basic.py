import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredHTMLLoader, PyPDFLoader, WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# Usando un Chat Model
chatModel = ChatOpenAI(model="gpt-4o-mini")

base_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(base_dir, "data/be-good.txt")
# file_path_csv = os.path.join(base_dir, "data/file_example_XLS.csv")
# file_path_html = os.path.join(base_dir, "data/100-startups.html")
# file_path_pdf = os.path.join(base_dir, "data/5pages.pdf")
# file_path_wikipedia = os.path.join(base_dir, "data/5pages.pdf")


# loader = TextLoader(file_path)
# loader_csv = CSVLoader(file_path_csv, encoding="ISO-8859-1")
# loader_html = CSVLoader(file_path_html)
# loader_pdf = PyPDFLoader(file_path_pdf) # poetry add pydf
loader_wikipedia = WikipediaLoader('query=name, load_max_docs=1') # poetry add pydf

# loaded_data = loader.load()
# loaded_data_csv = loader_csv.load()
# loaded_data_html = loader_html.load()
# loaded_data_pdf = loader_pdf.load_and_split()

chat_template = ChatPromptTemplate.from_messages(
    [
        ('human', 'Answer this {question}, here is some extra {context}')
    ]
)

messages =chat_template.format_messages(
    name='JFK',
    question='Donde nación JKF',
    context=loader_wikipedia
)

response = chatModel.invoke(messages)

print(response)
