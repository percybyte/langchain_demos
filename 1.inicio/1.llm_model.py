import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# Usando un Chat Model
llmModel = OpenAI()

# response = llmModel.invoke(
#     "Menciona el lenguaje de programación más popular"
# )

for chunk in llmModel.stream(
    "Dime el segundo lenguaje de programación más  utilizado"
):
    print(chunk, end="", flush=True)

# print(response)
