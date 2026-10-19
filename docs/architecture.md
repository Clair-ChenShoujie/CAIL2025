# 实现框架

## 数据层

读取 `0_train.json` 与 `1_train.json`，将答案统一成有序选项字符串。使用题干和选项计算 SHA-256 指纹，合并重复题目，将答案冲突题目写入 `rejected.jsonl`。与官方测试题相同的题目从训练候选中移除。

按题目类型、学科和单/多选分组，使用固定 seed 的指纹排序划分 train/dev。IFD 与 CoT 在划分之后处理训练数据。

## IFD

设题目指令为 x，标准答案为 y：

```text
L_cond   = mean CE(y | x)
L_uncond = mean CE(y)
IFD      = L_cond / L_uncond
```

两次损失使用同一组答案 token。条件损失的上下文使用 Qwen3 chat template，无条件损失从 BOS/EOS 上下文开始。仅对答案位置计算损失，排除超长样本。筛选区间内按 IFD 降序保留指定数量。

## CoT

Qwen-Max 接收题目、选项、标准答案与三段式模板，生成法理分析、逐项排查、结论推导。校验三段顺序、逐项分析、答案格式和标准答案一致性。标准答案只进入教师请求；学生 prompt 由原题重新生成。

生成记录携带模型、配置和消息内容的摘要，用于断点续跑。HTTP 暂时失败采用指数退避；结构校验失败重新生成。

## SFT

Qwen3-8B 使用 4bit NF4 + double quant，LoRA 注入 attention 和 MLP 的线性层。输入拼接 system、user、assistant，prompt 和 padding 的 label 为 `-100`。训练保存适配器，再合并成完整模型作为 GRPO 初始 actor。

## GRPO

使用 OpenRLHF 0.8.0 的 `train_ppo_ray`，配置 `advantage_estimator=group_norm`、`n_samples_per_prompt=8`。训练输入为已渲染的 Qwen3 prompt，使用 `enable_thinking=False` 对应的前缀。label 单独传给奖励回调。

```text
R = 0.6 × exact + 0.3 × set_F1 + 0.1 × format - 0.1 × extra
extra = 错选选项数 / 4
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

无法解析答案时奖励为 -1。`format` 表示三段式结构完整；`exact` 表示答案集合完全相等。非零的组内奖励差异为部分正确的回复提供学习信号。组内标准差使用样本标准差，与 OpenRLHF 0.8.0 对齐。

PPO clipping、组内归一化、参考模型 KL 和反向传播由 OpenRLHF 执行。`cail.rewards.group_advantages` 提供独立的组内计算函数。第一轮从合并后的 SFT 模型开始，第二轮从第一轮模型开始。

## 评测

推理使用固定 prompt，生成文本单独保存。评测从 `<answer>` 解析答案，严格准确率要求选项集合完全一致。

```text
set_F1 = 2 × |prediction ∩ gold| / (|prediction| + |gold|)
micro_F1 = 2TP / (2TP + FP + FN)
```

`set_f1` 为每题集合 F1 的宏平均，`micro_f1` 为所有选项合并后的 F1。`partial_correct_among_errors` 的分子为漏选、多选或部分重叠题数，分母为全部错误题数。

评测数据使用带答案的 dev。官方 test 用于生成答案映射。
