"""数据管道：加载文档 → 分块 → 向量化 → 构建 FAISS 索引并持久化。

运行：python src/ingest.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def load_documents(docs_dir):
    """读取 docs_dir 下所有 .md/.txt，返回 [{source, text}]。"""
    docs = []
    for fp in sorted(Path(docs_dir).glob("*.md")) + sorted(Path(docs_dir).glob("*.txt")):
        docs.append({"source": fp.name, "text": fp.read_text(encoding="utf-8")})
    return docs


def chunk_text(text, source, chunk_size, overlap):
    """滑动窗口分块：每块 chunk_size 字符，相邻块重叠 overlap，避免切断语义。"""
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append({"text": piece, "source": source})
        if end >= n:
            break
        start = end - overlap
    return chunks


def main():
    print(f"加载向量模型：{config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)

    docs = load_documents(config.DOCS_DIR)
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d["text"], d["source"], config.CHUNK_SIZE, config.CHUNK_OVERLAP))

    texts = [c["text"] for c in chunks]
    print(f"向量化 {len(texts)} 个分块 ...")
    vectors = model.encode(texts, normalize_embeddings=True)  # 归一化 → 内积即余弦相似度

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = 内积，配合归一化向量 = 余弦相似度
    index.add(np.asarray(vectors, dtype="float32"))

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    with open(config.INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"完成：{len(chunks)} 个分块已写入 {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
