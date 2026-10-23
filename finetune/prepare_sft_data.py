"""把 sft_data.json 转成 HuggingFace Dataset，并导出 jsonl 便于查看/复用。"""
import json
from pathlib import Path

from datasets import Dataset


def main():
    src = Path(__file__).parent / "sft_data.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    ds = Dataset.from_list(raw)
    print(f"SFT 数据条数：{len(ds)}")
    print(f"样例：{ds[0]}")

    out = Path(__file__).parent / "sft_data.jsonl"
    ds.to_json(str(out), force_ascii=False)
    print(f"已导出：{out}")


if __name__ == "__main__":
    main()
