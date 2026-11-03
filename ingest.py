import truststore
truststore.inject_into_ssl()
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
load_dotenv()

reader = SimpleDirectoryReader("data")
docs = reader.load_data()

# print(len(docs))

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(docs)

# print("my--long")
# print(len(nodes))
# print("my node text")
# print(nodes[10].text)
# print("my medadata text")
# print(nodes[10].metadata)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
index = VectorStoreIndex(nodes, show_progress=True)
index.storage_context.persist(persist_dir="storage")
print("האינדקס נשמר")
