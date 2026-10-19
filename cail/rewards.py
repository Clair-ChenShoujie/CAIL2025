import math

from .answers import normalize_answer, parse_answer, set_f1
from .prompts import cot_valid


def response_only(query, prompt):
    if query.startswith(prompt):
        return query[len(prompt):]
    marker = "<|im_start|>assistant\n"
    if marker in query:
        return query.rsplit(marker, 1)[1]
    raise ValueError("无法从 query 中定位 assistant 回复")


def reward_breakdown(response, gold):
    # 移除解码保留的结束标记。
    for token in ("<|im_end|>", "<|endoftext|>"):
        response = response.replace(token, "")
    gold = normalize_answer(gold)
    prediction = parse_answer(response)
    if prediction is None:
        return {"reward": -1.0, "exact": 0.0, "f1": 0.0, "format": 0.0, "extra": 0.0}
    exact = float(prediction == gold)
    f1 = set_f1(prediction, gold)
    formatted = float(cot_valid(response, prediction))
    extra = len(set(prediction) - set(gold)) / 4
    value = 0.6 * exact + 0.3 * f1 + 0.1 * formatted - 0.1 * extra
    return {"reward": value, "exact": exact, "f1": f1, "format": formatted, "extra": extra}


def group_advantages(rewards, group_size=8, epsilon=1e-8):
    if group_size < 2 or len(rewards) % group_size:
        raise ValueError("奖励数量应为组大小的整数倍，组大小至少为 2")
    if not all(math.isfinite(value) for value in rewards):
        raise ValueError("奖励必须是有限数")
    result = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start:start + group_size]
        mean = sum(group) / group_size
        std = math.sqrt(sum((value - mean) ** 2 for value in group) / (group_size - 1))
        result.extend((value - mean) / (std + epsilon) for value in group)
    return result


def reward_func(queries, prompts, labels, **kwargs):
    import torch
    if len(queries) != len(prompts) or len(queries) != len(labels):
        raise ValueError("query、prompt 和 label 数量不一致")
    values = [reward_breakdown(response_only(query, prompt), label)["reward"]
              for query, prompt, label in zip(queries, prompts, labels)]
    return torch.tensor(values, dtype=torch.float32)
