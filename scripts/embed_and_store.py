# FAISS/Graph building script placeholder
import os
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import networkx as nx

# --- 1. Configuration ---
KNOWLEDGE_BASE_DIR = "data/medical_pdfs/" # Folder containing your medical PDF documents
EMBEDDING_MODEL_NAME = "model/nomic-embed-text-v1"         # A powerful embedding model
FAISS_INDEX_PATH = "model/faiss_index"    # Path to save the FAISS index
GRAPH_PATH = "model/knowledge_graph.pkl"  # Path to save the knowledge graph

# --- 2. Load and Split Documents ---
print("Loading documents...")
# Assumes you have PDF files in the specified directory
loader = PyPDFDirectoryLoader(KNOWLEDGE_BASE_DIR)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# --- 3. Initialize Embedding Model ---
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
# Use 'cuda' to leverage the GPU for faster embedding
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cuda', 'trust_remote_code': True} # trust_remote_code=True allows custom code execution in the model
)

# --- 4. Create and Save FAISS Index ---
print("Creating and saving FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"FAISS index saved to {FAISS_INDEX_PATH}")

# --- 5. Create and Save Knowledge Graph (Simplified) ---
# A simple strategy: connect adjacent chunks from the same source document.
print("Creating knowledge graph...")
G = nx.Graph()

# Add nodes with content and metadata
for i, chunk in enumerate(chunks):
    source_file = os.path.basename(chunk.metadata.get("source", "Unknown"))
    node_id = f"chunk_{i}"
    G.add_node(node_id, content=chunk.page_content, source=source_file)

# Add edges to connect chunks from the same document
source_to_chunks = {}
for i, chunk in enumerate(chunks):
    source_file = os.path.basename(chunk.metadata.get("source", "Unknown"))
    if source_file not in source_to_chunks:
        source_to_chunks[source_file] = []
    source_to_chunks[source_file].append(f"chunk_{i}")

for source in source_to_chunks:
    nodes_in_source = source_to_chunks[source]
    for i in range(len(nodes_in_source) - 1):
        # Connect adjacent chunks
        G.add_edge(nodes_in_source[i], nodes_in_source[i+1])

with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)
print(f"Knowledge graph saved to {GRAPH_PATH}")

print("✅ Preprocessing complete!")