from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from langsmith import Client

# Use a local embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Test text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text("LangChain makes building LLM-powered apps easier.")
print("Chunks:", chunks)

# Test Chroma vectorstore (in-memory, with local embeddings)
vectorstore = Chroma.from_texts(["hello world"], embedding=embeddings)
print("Chroma collection:", vectorstore._collection.name)

# Test StateGraph
graph = StateGraph()
graph.add_node("start", lambda state: {"next": "end"})
graph.add_node("end", lambda state: {"result": "done"})
graph.set_entry_point("start")
app = graph.compile()
print("Graph result:", app.invoke({}))

# Test LangSmith client
client = Client()
print("LangSmith client initialized:", client)
