import concurrent.futures
import os
from pathlib import Path

from .api import ChatClient
from .io import append_jsonl, read_jsonl, unique_by_id
from .prompts import cot_valid, generation_key, teacher_messages


def generate_file(input_path, output_path, config, limit=None):
    api_key = os.environ.get(config.get("api_key_env", "DASHSCOPE_API_KEY"))
    if not api_key:
        raise ValueError(f"请设置环境变量 {config.get('api_key_env', 'DASHSCOPE_API_KEY')}")
    rows = list(unique_by_id(read_jsonl(input_path)).values())
    if limit is not None:
        rows = rows[:limit]
    temperature, max_tokens = config.get("temperature", 0.7), config.get("max_tokens", 2048)
    model, base_url = config["model"], config["base_url"]
    existing = list(read_jsonl(output_path)) if Path(output_path).exists() else []
    unique_by_id(existing)
    done = {row["generation_key"] for row in existing}
    current = {str(row["id"]): generation_key(row, model, base_url, temperature, max_tokens) for row in rows}
    for row in existing:
        if not cot_valid(row["cot"], row["answer"]):
            raise ValueError(f"已有推理链格式不正确: {row['id']}")
        key = str(row["id"])
        if key in current and current[key] != row["generation_key"]:
            raise ValueError("生成配置已改变，请使用新的输出文件")
    pending = [row for row in rows if current[str(row["id"])] not in done]
    client = ChatClient(base_url, model, api_key, timeout=config.get("timeout", 120), retries=config.get("retries", 4))

    def generate(row):
        for _ in range(config.get("validation_attempts", 3)):
            text = client.complete(teacher_messages(row), temperature=temperature, max_tokens=max_tokens)
            if cot_valid(text, row["answer"]):
                return {**row, "cot": text, "teacher": model, "generation_key": current[str(row["id"])]}
        raise ValueError("三段式推理链或答案校验未通过")

    successes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.get("workers", 4)) as executor:
        # 每批只提交固定数量的请求。
        width = config.get("workers", 4) * 2
        for start in range(0, len(pending), width):
            futures = {executor.submit(generate, row): row for row in pending[start:start + width]}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    append_jsonl(output_path, future.result())
                    successes += 1
                except Exception as exc:
                    append_jsonl(str(output_path) + ".errors.jsonl", {"id": row["id"], "reason": str(exc)})
    return {"resumed": len(rows) - len(pending), "generated": successes, "failed": len(pending) - successes}
