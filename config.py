"""全局配置：集中管理路径、模型、分块与检索参数。

所有可调参数都在这里，改一处全局生效，方便做效果对比实验（对应 JD 里的
"数据分析、效果评估、问题定位与持续优化"）。
"""
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 数据与索引
DOCS_DIR = BASE_DIR / "data" / "faq_docs"        # 原始 FAQ 文档（.md/.txt）
INDEX_DIR = BASE_DIR / "data" / "index"          # FAISS 索引与分块持久化
EVAL_SET = BASE_DIR / "data" / "eval_set.json"   # 评测集

# 向量模型（中文语义检索，开源、本地、无需 API）
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512

# 本地大模型（Ollama 推理）
LLM_MODEL = "qwen2.5:3b"                 # Ollama 里的模型名（3b 可稳定跑在 8GB 显存）
LLM_BASE_URL = "http://localhost:11434"  # Ollama 默认地址

# 分块参数
CHUNK_SIZE = 300      # 每个分块的最大字符数
CHUNK_OVERLAP = 50    # 相邻分块重叠字符数（避免切断语义）

# 检索参数
TOP_K = 4             # 最终喂给生成器的分块数
DENSE_K = 10          # 语义检索候选数
SPARSE_K = 10         # BM25 稀疏检索候选数
RRF_K = 60            # Reciprocal Rank Fusion 常数

# 生成参数
TEMPERATURE = 0.2     # 低温度 → 答案更稳定、更贴合资料
MAX_TOKENS = 512
