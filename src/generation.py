"""生成模块：Prompt 工程 —— 系统提示词 + 上下文注入 + 引用溯源要求。"""

SYSTEM_PROMPT = """你是「智联协同」企业产品的智能客服助手。请严格依据下面提供的【参考资料】回答用户问题。

规则：
1. 只依据参考资料回答，不要编造资料中没有的信息。
2. 若资料不足以回答，请明确回复「根据现有资料无法确定」。
3. 回答末尾用「[来源：文档名]」标注引用了哪些资料。
4. 用简洁、专业的中文回答。"""


def build_prompt(question, chunks):
    """把检索到的分块拼进 prompt，作为上下文注入。"""
    context = "\n\n".join(
        f"【参考资料{i + 1}】（来源：{c['source']}）\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"用户问题：{question}\n\n{context}\n\n请回答："
