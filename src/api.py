"""FastAPI 后端：对外提供 /query 问答接口与 /health 健康检查。

运行：uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import RAGPipeline

app = FastAPI(title="企业FAQ智能问答系统", description="RAG 检索增强问答 API")
pipeline = RAGPipeline()


class Query(BaseModel):
    question: str
    top_k: int = 4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(q: Query):
    return pipeline.query(q.question, top_k=q.top_k)
