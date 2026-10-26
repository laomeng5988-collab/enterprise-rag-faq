"""加载微调后的 LoRA 权重做推理测试，验证微调效果。"""
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA = str(Path(__file__).parent / "lora-output")


def load(use_lora=True):
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="auto")
    if use_lora:
        model = PeftModel.from_pretrained(model, LORA)
    model.eval()
    return tokenizer, model


def ask(tokenizer, model, q, max_new_tokens=60):
    text = (
        "<|im_start|>system\n你是一个企业产品客服助手。<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("assistant\n")[-1].strip()


if __name__ == "__main__":
    questions = [
        "专业版每成员每月多少钱？",
        "忘记密码怎么办？",
        "视频会议最多支持多少人？",
    ]
    for use_lora in (False, True):
        tag = "微调后" if use_lora else "基础模型"
        tok, mdl = load(use_lora)
        print(f"========== {tag} ==========")
        for q in questions:
            print(f"Q: {q}")
            print(f"A: {ask(tok, mdl, q)}\n")
        # 释放显存，避免加载两次模型 OOM
        del mdl
        torch.cuda.empty_cache()
