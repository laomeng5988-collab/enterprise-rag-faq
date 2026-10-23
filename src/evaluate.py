"""评估模块：检索召回率 recall@k + 答案命中率（top-1 分块关键词覆盖率）。

运行：python src/evaluate.py
"""
import sys
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from retrieval import HybridRetriever, tokenize


def recall_at_k(retriever, question, ref_docs, k):
    """检索 top-k 分块中，命中任一参考文档即算召回成功，返回 0/1。"""
    chunks = retriever.retrieve(question, top_k=k)
    sources = {c["source"] for c in chunks}
    return 1.0 if any(ref in sources for ref in ref_docs) else 0.0


def answer_hit_rate(retriever, question, ground_truth):
    """答案命中率（粗略）：top-1 分块中覆盖参考答案关键词的比例。"""
    top_chunks = retriever.retrieve(question, top_k=1)
    if not top_chunks:
        return 0.0
    gt_tokens = set(tokenize(ground_truth))
    if not gt_tokens:
        return 0.0
    hit_tokens = set(tokenize(top_chunks[0]["text"]))
    return len(gt_tokens & hit_tokens) / len(gt_tokens)


def main():
    embedder = SentenceTransformer(config.EMBED_MODEL)
    index = faiss.read_index(str(config.INDEX_DIR / "faiss.index"))
    chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    retriever = HybridRetriever(index, chunks, embedder.encode)

    eval_set = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))

    ks = [1, 4, 10]
    recalls = {k: [] for k in ks}
    hit_rates = []

    for item in eval_set:
        q, gt, refs = item["question"], item["ground_truth"], item["ref_docs"]
        for k in ks:
            recalls[k].append(recall_at_k(retriever, q, refs, k))
        hit_rates.append(answer_hit_rate(retriever, q, gt))

    print("===== 检索评估结果 =====")
    for k in ks:
        print(f"  recall@{k}: {sum(recalls[k]) / len(recalls[k]):.2%}")
    print(f"  答案命中率(top-1 关键词覆盖): {sum(hit_rates) / len(hit_rates):.2%}")
    print(f"  评测样本数: {len(eval_set)}")


if __name__ == "__main__":
    main()
