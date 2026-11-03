import truststore
truststore.inject_into_ssl()
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

# print(load_dotenv())

load_dotenv()
llm = GoogleGenAI(model="gemini-3.5-flash")
response = llm.complete("תגיד שלום בעברית במשפט אחד")
print(response.text)

