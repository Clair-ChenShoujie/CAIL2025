import math
from pathlib import Path

from .io import append_jsonl, digest, read_jsonl, unique_by_id, write_json, write_jsonl
from .prompts import messages


def ifd_ratio(conditional_loss, unconditional_loss):
    if not all(math.isfinite(x) and x > 0 for x in (conditional_loss, unconditional_loss)):
        raise ValueError("IFD 损失必须是正有限数")
    return conditional_loss / unconditional_loss


def score_file(input_path, output_path, model_path, max_length=2048, quantization="none", dtype="auto", limit=None):
    import torch
    import torch.nn.functional as F
    from .modeling import chat_text, load_model

    model, tokenizer = load_model(model_path, quantization=quantization, dtype=dtype)
    model.eval()
    signature = digest([str(model_path), max_length, quantization, dtype])
    previous = list(read_jsonl(output_path)) if Path(output_path).exists() else []
    previous_by_id = unique_by_id(previous)
    if any(row.get("run_signature") != signature for row in previous):
        raise ValueError("IFD 配置已改变，请使用新的输出文件")
    done = {row["score_key"] for row in previous}

    def answer_loss(prefix, target):
        if len(prefix) + len(target) > max_length:
            raise ValueError("超过 max_length")
        ids = torch.tensor([prefix + target], device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids).logits[0, len(prefix) - 1:-1].float()
            loss = F.cross_entropy(logits, ids[0, len(prefix):], reduction="mean")
        return float(loss.cpu())

    rows = list(unique_by_id(read_jsonl(input_path)).values())
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        target_text = f"<answer>{row['answer']}</answer>"
        prompt = chat_text(tokenizer, messages(row))
        key = digest([row, str(model_path), max_length, quantization, dtype, prompt, target_text])
        if key in done:
            continue
        if str(row["id"]) in previous_by_id:
            raise ValueError("IFD 输入已改变，请使用新的输出文件")
        target = tokenizer.encode(target_text, add_special_tokens=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        try:
            conditioned = answer_loss(prefix, target)
            unconditioned = answer_loss([bos], target)
            record = {"id": row["id"], "fingerprint": row["fingerprint"], "score_key": key,
                      "conditional_loss": conditioned, "unconditional_loss": unconditioned,
                      "ifd": ifd_ratio(conditioned, unconditioned), "model": str(model_path),
                      "run_signature": signature, "status": "ok"}
            append_jsonl(output_path, record)
        except ValueError as exc:
            append_jsonl(output_path, {"id": row["id"], "fingerprint": row["fingerprint"], "score_key": key,
                                       "run_signature": signature, "status": "excluded", "reason": str(exc)})


def select_file(input_path, scores_path, output_path, keep_ratio=18000 / 21000, max_ifd=1.0, keep_count=None):
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio 应在 0 与 1 之间")
    rows = unique_by_id(read_jsonl(input_path))
    scores = unique_by_id(read_jsonl(scores_path))
    selected = []
    for key, row in rows.items():
        score = scores.get(key)
        if score is None:
            raise ValueError(f"缺少 IFD 分数: {key}")
        if score["fingerprint"] != row["fingerprint"]:
            raise ValueError(f"题目与 IFD 分数不匹配: {key}")
        if score.get("status") == "excluded":
            continue
        value = score["ifd"]
        if math.isfinite(value) and 0 < value < max_ifd:
            selected.append({**row, "ifd": value})
    selected.sort(key=lambda row: (-row["ifd"], row["id"]))
    count = keep_count if keep_count is not None else (max(1, math.floor(len(rows) * keep_ratio)) if rows else 0)
    if count < 0:
        raise ValueError("keep_count 应大于等于 0")
    selected = selected[:count]
    write_jsonl(output_path, selected)
    report = {"input": len(rows), "eligible_and_selected": len(selected), "requested": count,
              "keep_ratio": keep_ratio, "max_ifd": max_ifd}
    write_json(str(output_path) + ".report.json", report)
    return report
