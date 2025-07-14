# GraphRAG pipeline logic placeholder
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import VLLM

# --- 1. Configuration ---
FAISS_INDEX_PATH = "model/faiss_index"
GRAPH_PATH = "model/knowledge_graph.pkl"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"  
FINETUNED_LLM_PATH = "model/merged_model" # Path to your fine-tuned medical QA model

def build_rag_chain():
    print("Loading all components...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cuda', 'trust_remote_code': True})
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = VLLM(
        model=FINETUNED_LLM_PATH,
        trust_remote_code=True,
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.7,
        tensor_parallel_size=1
    )
    print("Components loaded.")
    
    
    # --- 2.  Define Graph-Augmented Retrieval Logic ---
    def get_retrieved_docs(question: str):
        # Step A: Initial vector retrieval
        initial_docs = retriever.invoke(question)
        
        # Step B: Graph expansion
        retrieved_content = {doc.page_content for doc in initial_docs}
        for doc in initial_docs:
            # FAISS might store doc IDs differently, inspect 'doc.metadata' to find the key
            node_id = f"chunk_{doc.metadata.get('doc_id')}" # This key might need adjustment
            if G.has_node(node_id):
                for neighbor in G.neighbors(node_id):
                    if "content" in G.nodes[neighbor]:
                        retrieved_content.add(G.nodes[neighbor]['content'])

        return "\n\n---\n\n".join(list(retrieved_content))

    # --- 4. Define Prompt and Build RAG Chain ---
    template = """
    [SYSTEM INSTRUCTION]
    You are a professional medical question-answering assistant. Use the provided context below to answer the user's question in a professional, rigorous, and clear manner.
    Do not use any external knowledge not mentioned in the context. If the context is insufficient to answer the question, state that you cannot answer based on the provided information.

    [CONTEXT]
    {context}

    [USER QUESTION]
    {question}

    [YOUR ANSWER]
    """
    prompt = PromptTemplate.from_template(template)

    # Build the final chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {
            "context": RunnableLambda(get_retrieved_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Test (Optional) ---
if __name__ == "__main__":
    rag_chain = build_rag_chain()
    test_question = "消极的商品评论有哪些特征？"
    print(f"Testing with question: {test_question}")
    answer = rag_chain.invoke(test_question)
    print("\n[Generated Answer]")
    print(answer)