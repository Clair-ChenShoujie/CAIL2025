from pathlib import Path

from .io import read_jsonl, unique_by_id, write_jsonl
from .prompts import cot_valid, messages, qwen_prompt


def export_data(cot_path, train_path, dev_path, output_dir):
    train = unique_by_id(read_jsonl(train_path))
    dev = unique_by_id(read_jsonl(dev_path))
    if set(train) & set(dev) or {row["fingerprint"] for row in train.values()} & {row["fingerprint"] for row in dev.values()}:
        raise ValueError("训练集与验证集存在重叠")
    cot = unique_by_id(read_jsonl(cot_path))
    sft = []
    for key, row in cot.items():
        original = train.get(key)
        if original is None or original["fingerprint"] != row["fingerprint"] or original["answer"] != row["answer"]:
            raise ValueError(f"CoT 与训练数据不匹配: {key}")
        if not cot_valid(row["cot"], row["answer"]):
            raise ValueError(f"CoT 格式不正确: {key}")
        sft.append({"id": key, "messages": messages(original) + [{"role": "assistant", "content": row["cot"]}]})
    sft.sort(key=lambda row: row["id"])
    output = Path(output_dir)
    write_jsonl(output / "sft_train.jsonl", sft)
    # GRPO 标签单独保存。
    for name, values in (("grpo_train", train.values()), ("grpo_dev", dev.values())):
        write_jsonl(output / f"{name}.jsonl", ({"id": row["id"], "prompt": qwen_prompt(row),
                                               "label": row["answer"]} for row in values))
    return {"sft": len(sft), "grpo_train": len(train), "grpo_dev": len(dev)}
