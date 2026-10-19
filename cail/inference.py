import concurrent.futures
import os
from pathlib import Path

from .answers import parse_answer
from .io import append_jsonl, digest, read_jsonl, unique_by_id
from .prompts import messages


def infer_file(input_path, output_path, model_path, config, adapter=None, backend="local", limit=None):
    rows = list(unique_by_id(read_jsonl(input_path)).values())
    if limit is not None:
        rows = rows[:limit]
    signature = digest([str(model_path), str(adapter), backend, config])
    previous = list(read_jsonl(output_path)) if Path(output_path).exists() else []
    unique_by_id(previous)
    if any(row.get("run_signature") != signature for row in previous):
        raise ValueError("推理配置已改变，请使用新的输出文件")
    completed = {str(row["id"]): row for row in previous}
    for row in rows:
        old = completed.get(str(row["id"]))
        if old and old["question_digest"] != digest(messages(row)):
            raise ValueError("推理题目已改变，请使用新的输出文件")
    pending = [row for row in rows if str(row["id"]) not in completed]
    if not pending:
        return {"resumed": len(rows), "generated": 0}

    def save(row, text):
        append_jsonl(output_path, {"id": row["id"], "output": text, "prediction": parse_answer(text),
                                   "question_digest": digest(messages(row)), "run_signature": signature,
                                   "model": str(model_path)})

    if backend == "api":
        from .api import ChatClient
        key = os.environ.get(config.get("api_key_env", "MODEL_API_KEY"), "")
        client = ChatClient(config["base_url"], model_path, key, retries=config.get("retries", 4))
        def request(row):
            return client.complete(messages(row), config.get("temperature", 0), config.get("max_new_tokens", 2048))
        workers = config.get("workers", 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(pending), workers * 2):
                futures = {pool.submit(request, row): row for row in pending[start:start + workers * 2]}
                for future in concurrent.futures.as_completed(futures):
                    save(futures[future], future.result())
    else:
        import torch
        from transformers import set_seed
        from .modeling import chat_text, load_model
        set_seed(config.get("seed", 42))
        model, tokenizer = load_model(model_path, config.get("quantization", "none"), config.get("dtype", "auto"), adapter=adapter)
        model.eval()
        tokenizer.padding_side = "left"
        batch_size = config.get("batch_size", 1)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            prompts = [chat_text(tokenizer, messages(row)) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            if encoded.input_ids.shape[1] > config.get("prompt_max_length", 4096):
                raise ValueError("输入超过 prompt_max_length")
            temperature = config.get("temperature", 0)
            kwargs = {"do_sample": temperature > 0, "max_new_tokens": config.get("max_new_tokens", 2048),
                      "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=config.get("top_p", 0.95))
            with torch.inference_mode():
                generated = model.generate(**encoded, **kwargs)
            outputs = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
            for row, text in zip(batch, outputs):
                save(row, text)
    return {"resumed": len(rows) - len(pending), "generated": len(pending)}
