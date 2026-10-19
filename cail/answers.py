import re
import unicodedata

TAG = re.compile(r"<answer>\s*([^<>]*?)\s*</answer>", re.I)
LETTERS = set("ABCD")


def normalize_answer(value):
    if isinstance(value, (list, tuple, set)):
        value = "".join(str(item) for item in value)
    if not isinstance(value, str):
        raise ValueError("答案必须是字符串或选项列表")
    value = unicodedata.normalize("NFKC", value).upper().strip()
    value = re.sub(r"[\s,，、;；\[\]'\"]", "", value)
    if not value or any(letter not in LETTERS for letter in value):
        raise ValueError(f"无效答案: {value!r}")
    if len(set(value)) != len(value):
        raise ValueError("答案含重复选项")
    return "".join(sorted(value))


def parse_answer(text):
    if not isinstance(text, str):
        return None
    matches = TAG.findall(text)
    if len(matches) != 1:
        return None
    try:
        return normalize_answer(matches[0])
    except ValueError:
        return None


def set_f1(prediction, gold):
    p, g = set(prediction or ""), set(normalize_answer(gold))
    return 2 * len(p & g) / (len(p) + len(g)) if p else 0.0


def error_type(prediction, gold):
    if prediction is None:
        return "invalid_format"
    p, g = set(prediction), set(normalize_answer(gold))
    if p == g:
        return "correct"
    if p < g:
        return "missing_options"
    if g < p:
        return "extra_options"
    return "partial_overlap" if p & g else "fully_wrong"
