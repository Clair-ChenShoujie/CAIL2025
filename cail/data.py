import collections
import math
import re
import unicodedata
from pathlib import Path

from .answers import normalize_answer
from .io import digest, read_jsonl, write_json, write_jsonl


def clean_text(value):
    if not isinstance(value, str):
        raise ValueError("文本字段必须是字符串")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def normalize_row(row, labeled=True):
    if not isinstance(row, dict):
        raise ValueError("记录必须是对象")
    result = {"id": str(row.get("id", "")).strip(), "statement": clean_text(row.get("statement", "")),
              "subject": clean_text(row.get("subject", "未知")), "type": str(row.get("type", "未知"))}
    options = row.get("option_list", {})
    if not result["id"] or not result["statement"] or not isinstance(options, dict) or set(options) != set("ABCD"):
        raise ValueError("id、题干或 A-D 选项不完整")
    result["option_list"] = {key: clean_text(options[key]) for key in "ABCD"}
    if not all(result["option_list"].values()):
        raise ValueError("选项为空")
    if labeled:
        result["answer"] = normalize_answer(row["answer"])
    # 指纹使用题干与选项内容。
    result["fingerprint"] = digest({"statement": result["statement"], "options": result["option_list"]})
    return result


def prepare(source, output, dev_ratio=0.1, seed=42):
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio 应在 0 与 1 之间")
    source, output = Path(source), Path(output)
    train_paths = [source / f"{kind}_train.json" for kind in (0, 1)]
    test_paths = [source / f"{kind}_test.json" for kind in (0, 1)]
    rows, tests, rejected = [], [], []
    for labeled, paths, target in ((True, train_paths, rows), (False, test_paths, tests)):
        for path in paths:
            for number, raw in enumerate(read_jsonl(path), 1):
                try:
                    target.append(normalize_row(raw, labeled))
                except (ValueError, KeyError, TypeError) as exc:
                    rejected.append({"file": path.name, "line": number, "reason": str(exc)})
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["fingerprint"]].append(row)
    test_fingerprints = {row["fingerprint"] for row in tests}
    unique, duplicates, conflicts, overlap = [], 0, 0, 0
    for fp, group in sorted(groups.items()):
        if len({row["answer"] for row in group}) > 1:
            conflicts += len(group)
            rejected.extend({"id": row["id"], "reason": "同题答案冲突"} for row in group)
        elif fp in test_fingerprints:
            overlap += len(group)
        else:
            unique.append(sorted(group, key=lambda row: row["id"])[0])
            duplicates += len(group) - 1
    ids = [row["id"] for row in unique]
    if len(ids) != len(set(ids)):
        raise ValueError("不同题目使用了相同 id")
    buckets = collections.defaultdict(list)
    for row in unique:
        buckets[(row["type"], row["subject"], len(row["answer"]) > 1)].append(row)
    train, dev = [], []
    for key, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda row: digest([seed, row["fingerprint"]]))
        count = min(len(bucket) - 1, max(1, math.floor(len(bucket) * dev_ratio))) if len(bucket) > 1 else 0
        dev.extend(bucket[:count])
        train.extend(bucket[count:])
    train.sort(key=lambda row: row["id"])
    dev.sort(key=lambda row: row["id"])
    for name, values in (("train", train), ("dev", dev), ("test", tests), ("rejected", rejected)):
        write_jsonl(output / f"{name}.jsonl", values)
    report = {"raw_train": len(rows), "raw_test": len(tests), "train": len(train), "dev": len(dev),
              "duplicates": duplicates, "conflicting_rows": conflicts, "test_overlap_removed": overlap,
              "invalid_rows": sum("file" in row for row in rejected), "seed": seed, "dev_ratio": dev_ratio,
              "train_digest": digest(train), "dev_digest": digest(dev)}
    write_json(output / "prepare_report.json", report)
    return report
