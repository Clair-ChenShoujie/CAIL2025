# 数据格式

## 处理后的题目

```json
{"id":"q1","statement":"题干","option_list":{"A":"选项一","B":"选项二","C":"选项三","D":"选项四"},"answer":"AB","subject":"民法","type":"0","fingerprint":"sha256"}
```

`test.jsonl` 不含 `answer`。

## IFD 评分

```json
{"id":"q1","fingerprint":"sha256","score_key":"sha256","run_signature":"sha256","conditional_loss":0.5,"unconditional_loss":1.0,"ifd":0.5,"model":"模型路径","status":"ok"}
```

超长样本使用 `status=excluded` 并保存 `reason`。

## CoT

在题目字段上增加 `cot`、`teacher` 和 `generation_key`。

## SFT

```json
{"id":"q1","messages":[{"role":"system","content":"系统提示"},{"role":"user","content":"题干与选项"},{"role":"assistant","content":"三段式推理与 <answer>AB</answer>"}]}
```

## GRPO

```json
{"id":"q1","prompt":"<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","label":"AB"}
```

## 推理记录

```json
{"id":"q1","output":"生成文本 <answer>AB</answer>","prediction":"AB","question_digest":"sha256","run_signature":"sha256","model":"模型路径或服务名"}
```

## 答案映射

```json
{"q1":["A","B"],"q2":["D"]}
```
