"""LoRA 指令微调：对 Qwen2.5-1.5B-Instruct 做轻量微调（SFT）。

运行：python finetune/train_lora.py
要求：NVIDIA GPU（约 8GB 显存即可，FP16 全精度微调，无需 bitsandbytes）。
微调后得到 LoRA 权重，可用 transformers + peft 加载后做推理，或合并回基座模型。
"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = str(Path(__file__).parent / "lora-output")
MAX_LENGTH = 512


def format_example(ex):
    """Qwen ChatML 格式：system / user / assistant。"""
    return (
        f"<|im_start|>system\n你是一个企业产品客服助手。<|im_end|>\n"
        f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
    )


def tokenize_fn(ex, tokenizer):
    text = format_example(ex)
    out = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    out["labels"] = out["input_ids"].copy()  # 自回归：labels 即 input_ids
    return out


def main():
    data = json.loads((Path(__file__).parent / "sft_data.json").read_text(encoding="utf-8"))
    ds = Dataset.from_list(data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = ds.map(lambda ex: tokenize_fn(ex, tokenizer), remove_columns=ds.column_names)

    use_cuda = torch.cuda.is_available()
    print(f"CUDA 可用: {use_cuda}（{'GPU' if use_cuda else 'CPU'} 训练）")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=use_cuda,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"微调完成，LoRA 权重已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
