import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser


from langchain_core.pydantic_v1 import BaseModel, Field

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# Inicializa el modelo de chat de OpenAI con el modelo especificado
chatModel = ChatOpenAI(model="gpt-4o-mini")


# Creamos nuestro template
json_prompt = PromptTemplate.from_template(
    "Retorna un objeto JSON con la llave `clave` para responder la siguiente instruccion: {question}"
)

json_parser = SimpleJsonOutputParser()

# Construimos la cadena
json_chain = json_prompt | chatModel | json_parser

# response = json_chain.invoke({"question": "Actual presidente del Perú"})


# Usando Pydantic
class Joke(BaseModel):
    setup: str = Field(description='question to set up a joke')
    punchline: str = Field(description='answer to resolve a joke')


# Definir el parse basandonos en el objeto de Pydantic
parser = PydanticOutputParser(pydantic_object=Joke)

# Agregamos el parser en la definición del prompt
prompt = PromptTemplate(
    template="Anwer the use query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Creamos la cadena
chain = prompt | chatModel | parser

response = chain.invoke({'query': 'Tell me a joke'})
print(response)
