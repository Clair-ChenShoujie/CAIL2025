from .answers import normalize_answer, parse_answer
from .io import digest

SYSTEM = "你是中国法律考试解题助手。依据题干与选项进行推理，最后用 <answer>选项字母</answer> 输出答案。"
HEADINGS = ("【法理分析】", "【逐项排查】", "【结论推导】")


def question_text(row):
    options = "\n".join(f"{key}. {row['option_list'][key]}" for key in "ABCD")
    return f"题目：{row['statement']}\n{options}"


def user_prompt(row):
    return (question_text(row) + "\n请按【法理分析】、【逐项排查】、【结论推导】三段作答。"
            "逐项分析 A、B、C、D，选出所有符合题意的选项。"
            "最后一行输出 <answer>AB</answer>，字母按顺序排列。")


def messages(row):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(row)}]


def qwen_prompt(row):
    text = "".join(f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages(row))
    return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def teacher_messages(row):
    prompt = user_prompt(row) + f"\n标准答案：{normalize_answer(row['answer'])}。请给出与标准答案一致的完整推理。"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def cot_valid(text, answer):
    if parse_answer(text) != normalize_answer(answer):
        return False
    positions = [text.find(heading) for heading in HEADINGS]
    if min(positions) < 0 or positions != sorted(positions):
        return False
    for index, start in enumerate(positions):
        end = positions[index + 1] if index < 2 else text.find("<answer>")
        if end <= start or len(text[start + len(HEADINGS[index]):end].strip()) < 5:
            return False
    section = text[positions[1]:positions[2]]
    return all(letter in section for letter in "ABCD") and text.rstrip().endswith("</answer>")


def generation_key(row, model, base_url, temperature, max_tokens):
    return digest({"messages": teacher_messages(row), "model": model, "base_url": base_url,
                   "temperature": temperature, "max_tokens": max_tokens})
