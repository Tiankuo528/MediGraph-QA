# FastAPI serving script placeholder
from fastapi import FastAPI
from pydantic import BaseModel
from scripts.rag_chain import build_rag_chain
from contextlib import asynccontextmanager

app = FastAPI(
    title="Medical QA RAG API",
    description="API for answering medical questions using a fine-tuned LLM with GraphRAG."
)

rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("Initializing RAG chain...")
    rag_chain = build_rag_chain()
    print("RAG chain ready.")
    yield

app.router.lifespan_context = lifespan

class QueryRequest(BaseModel):
    question: str

@app.post("/generate", summary="Generate an answer to a medical question")
async def generate_answer(request: QueryRequest):
    if rag_chain is None:
        return {"error": "Model not ready"}
    answer = rag_chain.invoke(request.question)
    return {"question": request.question, "answer": answer}

@app.get("/")
async def root():
    return {"msg": "Medical QA RAG API is running. Use POST /generate to ask questions."}

