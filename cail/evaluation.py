import collections
from pathlib import Path

from .answers import error_type, normalize_answer, parse_answer, set_f1
from .io import read_jsonl, unique_by_id, write_json, write_jsonl, atomic_text


def summarize(records):
    count = len(records)
    errors = collections.Counter(row["error_type"] for row in records)
    tp = sum(len(set(row["prediction"] or "") & set(row["answer"])) for row in records)
    fp = sum(len(set(row["prediction"] or "") - set(row["answer"])) for row in records)
    fn = sum(len(set(row["answer"]) - set(row["prediction"] or "")) for row in records)
    incorrect = count - errors["correct"]
    partial = sum(errors[key] for key in ("missing_options", "extra_options", "partial_overlap"))
    return {"count": count, "exact_accuracy": errors["correct"] / count if count else 0.0,
            "set_f1": sum(row["f1"] for row in records) / count if count else 0.0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "invalid_format_rate": errors["invalid_format"] / count if count else 0.0,
            "partial_correct_among_errors": partial / incorrect if incorrect else 0.0,
            "errors": dict(errors)}


def evaluate(gold_path, predictions_path, output_dir, allow_missing=False):
    gold = unique_by_id(read_jsonl(gold_path))
    predictions = unique_by_id(read_jsonl(predictions_path))
    if set(predictions) - set(gold):
        raise ValueError("预测包含不属于评测集的 id")
    missing = set(gold) - set(predictions)
    if missing and not allow_missing:
        raise ValueError(f"缺少 {len(missing)} 条预测")
    records = []
    for key, row in gold.items():
        answer = normalize_answer(row["answer"])
        raw = predictions.get(key, {}).get("output", "")
        prediction = parse_answer(raw)
        records.append({"id": key, "statement": row["statement"], "option_list": row["option_list"],
                        "answer": answer, "prediction": prediction, "output": raw,
                        "f1": set_f1(prediction, answer), "error_type": error_type(prediction, answer),
                        "subject": row.get("subject", "未知"), "type": row.get("type", "未知"),
                        "multiple": len(answer) > 1})
    report = summarize(records)
    report["missing_predictions"] = len(missing)
    report["by_answer_count"] = {name: summarize([row for row in records if row["multiple"] == multiple])
                                 for name, multiple in (("single", False), ("multiple", True))}
    report["by_subject"] = {subject: summarize([row for row in records if row["subject"] == subject])
                            for subject in sorted({row["subject"] for row in records})}
    report["by_type"] = {kind: summarize([row for row in records if row["type"] == kind])
                         for kind in sorted({row["type"] for row in records})}
    output = Path(output_dir)
    write_json(output / "metrics.json", report)
    write_jsonl(output / "records.jsonl", records)
    write_jsonl(output / "bad_cases.jsonl", (row for row in records if row["error_type"] != "correct"))
    return report


def compare_runs(named_paths, output_path):
    import json
    rows = []
    for name, path in named_paths:
        report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        multi = report["by_answer_count"]["multiple"]
        rows.append((name, report["count"], report["exact_accuracy"], report["set_f1"],
                     multi["exact_accuracy"], multi["set_f1"], report["invalid_format_rate"]))
    lines = ["| 实验 | 题数 | 严格准确率 | 集合 F1 | 多选准确率 | 多选 F1 | 格式错误率 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, count, *values in rows:
        lines.append(f"| {name} | {count} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    atomic_text(output_path, "\n".join(lines) + "\n")


def submission(predictions_path, output_path):
    rows = list(unique_by_id(read_jsonl(predictions_path)).values())
    result = {}
    for row in rows:
        answer = parse_answer(row["output"])
        if answer is None:
            raise ValueError(f"答案格式不正确: {row['id']}")
        result[str(row["id"])] = list(answer)
    write_json(output_path, result)
    return {"count": len(result)}
