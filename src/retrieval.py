"""检索模块：语义检索（FAISS）+ BM25 稀疏检索 + RRF 融合。

混合检索思路：语义向量擅长「意思相近」，BM25 擅长「关键词精确命中」，
两者互补；用 Reciprocal Rank Fusion 把两路排序融合成一路。
"""
import numpy as np
import jieba
from rank_bm25 import BM25Okapi


def tokenize(text):
    """中文分词，供 BM25 使用。"""
    return [t for t in jieba.cut(text) if t.strip()]


class HybridRetriever:
    def __init__(self, index, chunks, embed_fn):
        self.index = index          # FAISS 索引
        self.chunks = chunks        # [{text, source}]，顺序与 FAISS 行号一一对应
        self.embed_fn = embed_fn    # 向量化函数（sentence-transformers encode）
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def dense_search(self, query, k):
        """语义检索：向量相似度 top-k。"""
        qv = self.embed_fn([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(qv, dtype="float32"), k)
        return list(zip(ids[0].tolist(), scores[0].tolist()))

    def sparse_search(self, query, k):
        """稀疏检索：BM25 关键词打分 top-k。"""
        scores = self.bm25.get_scores(tokenize(query))
        ids = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in ids]

    def rrf_fuse(self, dense_ranked, sparse_ranked, k=60):
        """Reciprocal Rank Fusion：rank 越靠前贡献越大，把两路合并。"""
        rrf = {}
        for rank, (doc_id, _) in enumerate(dense_ranked):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse_ranked):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query, top_k=4):
        dense = self.dense_search(query, k=10)
        sparse = self.sparse_search(query, k=10)
        fused = self.rrf_fuse(dense, sparse)
        return [self.chunks[doc_id] for doc_id, _ in fused[:top_k]]
