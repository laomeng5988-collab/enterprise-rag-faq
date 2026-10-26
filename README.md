# 企业产品 FAQ 智能问答系统（RAG）

基于 **RAG（检索增强生成）+ 本地大模型** 的企业产品 FAQ 智能客服系统。将产品文档/FAQ 通过「数据管道 → 向量检索 → Prompt 生成」完整链路实现知识库问答，并包含**检索效果评估**、**LoRA 微调脚本**与 **FastAPI 后端接口**。

> 全程本地运行：本地向量模型 + 本地大模型（Ollama），**无需任何 API Key**。

## 架构

```
文档(.md) ──→ 分块 ──→ 向量化(BGE) ──→ FAISS 索引
                                          ↓
用户问题 ──→ 混合检索(语义+BM25+RRF) ──→ Prompt 工程 ──→ Qwen(本地) ──→ 答案+引用
                                          ↓
                                  效果评估(recall@k)
```

## 核心模块

| 模块 | 说明 | 对应技能 |
|---|---|---|
| 数据管道 | 文档加载 → 滑动窗口分块 → 向量化 → FAISS 索引持久化 | 数据管道打通 |
| 混合检索 | 语义检索（FAISS）+ BM25 稀疏检索 + RRF 融合 | RAG 检索 |
| Prompt 工程 | 系统提示词 + 上下文注入 + 引用溯源要求 | Prompt 工程 |
| 效果评估 | 检索召回率 recall@k + 答案命中率 | 效果评估、问题定位 |
| LoRA 微调 | Qwen2.5 + PEFT 轻量指令微调（SFT） | 模型微调 |
| 后端接口 | FastAPI 提供 `/query` 问答接口 | 后端接口 |

## 技术栈

Python · sentence-transformers（BGE 向量化）· FAISS · BM25 · Ollama（Qwen2.5）· FastAPI · PEFT（LoRA）

## 快速开始

### 1. 环境准备

- 安装 [Ollama](https://ollama.com) 并拉取模型：`ollama pull qwen2.5:3b`
- 安装依赖：`pip install -r requirements.txt`

### 2. 构建索引（数据管道）

```bash
python src/ingest.py   # 下载 BGE 向量模型 + 构建 FAISS 索引
```

### 3. 检索评估

```bash
python src/evaluate.py
```

### 4. 问答（RAG 主流程）

```bash
python src/pipeline.py   # 交互式问答
```

### 5. 启动 API 服务

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
# POST /query  {"question": "专业版多少钱？"}
```

## 检索评估结果（示例数据集）

| 指标 | 数值 |
|---|---|
| recall@1 | 93.33% |
| recall@4 | 100% |
| recall@10 | 100% |

## LoRA 微调

```bash
python finetune/train_lora.py   # 对 Qwen2.5-1.5B 做 LoRA SFT（需 NVIDIA GPU）
python finetune/test_lora.py    # 微调前后对比推理测试
```

- 微调数据：`finetune/sft_data.json`（30 条 FAQ 指令对）
- 技术：PEFT LoRA（r=16），FP16 全精度微调，无需 bitsandbytes

### 微调结果

| 指标 | 数值 |
|---|---|
| 基座模型 | Qwen2.5-1.5B-Instruct |
| 训练数据 | 30 条 FAQ 指令对，3 epoch |
| 可训练参数 | 18.5M（占全量 1.18%）|
| Loss | 2.844 → 1.373 |
| 训练时长 | 26.5 秒（RTX 4070 GPU）|

**微调前后对比**：
- 基础模型：对产品定价等事实性问题倾向「拒答」或给出通用回答
- 微调后：能直接给出 FAQ 风格的简洁答案

**诚实说明**：30 条小样本下，微调模型对具体数值（价格、人数）仍可能出错。这正是 **RAG 与 SFT 的分工**——RAG 通过检索注入真实文档来保证「事实准确」，微调更适合学习「回答风格」而非「记忆事实」。两者互补，本项目同时实现，以对比验证这一边界。

## 目录结构

```
.
├── config.py               # 全局配置（模型/分块/检索参数）
├── data/
│   ├── faq_docs/           # 示例 FAQ 文档（4 篇）
│   └── eval_set.json       # 评测集（15 条问答）
├── src/
│   ├── ingest.py           # 数据管道
│   ├── retrieval.py        # 混合检索
│   ├── llm.py              # LLM 接口封装
│   ├── generation.py       # Prompt 工程
│   ├── pipeline.py         # RAG 主流程
│   ├── evaluate.py         # 效果评估
│   └── api.py              # FastAPI 后端
├── finetune/
│   ├── sft_data.json       # 微调数据
│   ├── prepare_sft_data.py # 数据预处理
│   └── train_lora.py       # LoRA 微调
└── requirements.txt
```

## 说明

- 示例数据为虚构企业产品「智联协同（ZhiLink）」的 FAQ，可替换为自己的文档。
- 向量模型与 LLM 均本地运行，不依赖外部 API，可离线复现。
- 混合检索采用 RRF（Reciprocal Rank Fusion）融合语义与关键词两路召回。
