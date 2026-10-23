"""RAG 主流程：加载索引 → 混合检索 → Prompt 生成 → 返回答案与来源。

交互式运行：python src/pipeline.py
"""
import sys
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from retrieval import HybridRetriever
from generation import build_prompt, SYSTEM_PROMPT
from llm import OllamaLLM


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.index = faiss.read_index(str(config.INDEX_DIR / "faiss.index"))
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.retriever = HybridRetriever(self.index, self.chunks, self.embedder.encode)
        self.llm = OllamaLLM(
            config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )

    def query(self, question, top_k=None):
        top_k = top_k or config.TOP_K
        chunks = self.retriever.retrieve(question, top_k=top_k)
        answer = self.llm.generate(build_prompt(question, chunks), system=SYSTEM_PROMPT)
        return {
            "question": question,
            "answer": answer,
            "sources": [c["source"] for c in chunks],
        }


if __name__ == "__main__":
    p = RAGPipeline()
    print("RAG 问答系统已就绪，输入问题开始（输入 q 退出）")
    while True:
        q = input("\n>>> ").strip()
        if q.lower() == "q":
            break
        r = p.query(q)
        print(f"\n回答：{r['answer']}")
        print(f"来源：{r['sources']}")
