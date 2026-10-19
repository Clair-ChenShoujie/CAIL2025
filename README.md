# CAIL2025

这是一个面向 CAIL2025 司法多选推理任务的项目实现，基于 JEC-QA 和 Qwen3-8B，串联 IFD 数据筛选、三段式 CoT 构建、QLoRA 监督微调、GRPO 强化微调以及推理评测。

项目提供各阶段的独立命令、统一配置和数据格式，可按数据处理 → SFT → GRPO → 评测的顺序使用。

```text
JEC-QA
  └─ 校验、去重、划分 ───────────────→ dev / test
       └─ train → IFD → selected
                         ├─ Qwen-Max → CoT → SFT → 合并权重
                         └─ GRPO prompts ──────────┤
                                                  └─ GRPO 第一轮 → 第二轮
各阶段模型 → 批量推理 → 严格准确率 / 集合 F1 / bad cases
```

## 目录

```text
cail/
  data.py          数据校验、去重、划分
  ifd.py           答案损失计算与 IFD 筛选
  api.py           Chat Completions 客户端
  cot.py           并发生成推理链、校验、续跑
  prompts.py       题目模板、三段式 CoT 模板
  export.py        SFT 与 GRPO 格式转换
  modeling.py      模型加载、量化、权重合并
  sft.py           QLoRA 训练、答案区域 loss mask
  rewards.py       多选奖励、组内 advantage
  grpo.py          OpenRLHF 命令构建与启动
  inference.py     本地/API 批量推理
  evaluation.py    评测、错误分析、实验对比
  cli.py           命令行入口
configs/           本地与 Linux 配置
scripts/           GRPO 启动脚本与奖励回调
tests/             核心逻辑测试
docs/              实现框架与数据格式
```

## 安装与配置

Python 3.10 / 3.11。SFT 环境安装：

```bash
python -m venv .venv
```

激活 `.venv`：PowerShell 使用 `.\.venv\Scripts\Activate.ps1`，Linux 使用 `source .venv/bin/activate`。然后安装依赖：

```bash
python -m pip install -e ".[train]"
```

`configs/pipeline.json` 中配置数据目录、模型目录、IFD、CoT、SFT、GRPO 和推理参数。默认数据与模型路径：

```text
I:/neu/01CAIL/JEC-QA
I:/neu/01CAIL/Qwen3-8B
```

Linux 修改 `configs/pipeline.linux.json` 中的路径，所有命令增加 `--config configs/pipeline.linux.json`，例如：

```bash
python -m cail --config configs/pipeline.linux.json prepare
```

相对的产物路径以项目根目录为基准。

## 数据处理与 IFD

```bash
python -m cail prepare
python -m cail ifd
python -m cail select
```

处理结果位于 `data/processed`：`train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `prepare_report.json`。

IFD 使用 `L(answer | instruction) / L(answer)`，仅计算答案 token 的平均交叉熵。筛选 `0 < IFD < 1` 的样本，按 IFD 从高到低保留。`selection.keep_count` 控制数量；设为 `null` 后使用 `keep_ratio`。评分保存在 `ifd_scores.jsonl`，筛选结果保存在 `selected.jsonl`。

## CoT 与训练数据

设置 Qwen-Max 密钥：

```powershell
$env:DASHSCOPE_API_KEY = "你的密钥"
```

```bash
export DASHSCOPE_API_KEY="你的密钥"
```

执行生成和转换：

```bash
python -m cail cot
python -m cail export
```

生成结果为 `cot.jsonl`；失败记录为 `cot.jsonl.errors.jsonl`。再次执行 `cot` 会跳过已完成的题目。`--limit 20` 可指定本次处理数量。

三段式输出：

```text
【法理分析】
分析相关法律规则。
【逐项排查】
A. ……
B. ……
C. ……
D. ……
【结论推导】
汇总符合题意的选项。
<answer>AB</answer>
```

`export` 输出 `sft_train.jsonl`、`grpo_train.jsonl` 和 `grpo_dev.jsonl`。

## QLoRA SFT

```bash
python -m cail sft
python -m cail merge
```

SFT 使用 NF4、双重量化和 LoRA；只对 assistant 的推理与答案计算 loss。超长样本记录到 `training_summary.json`。

```bash
python -m cail sft --resume checkpoints/sft-adapter/checkpoint-100
```

适配器保存到 `checkpoints/sft-adapter`，合并模型保存到 `checkpoints/sft-merged`。

## GRPO

在 Linux CUDA 环境安装独立的 GRPO 环境：

```bash
python -m pip install -e .
python -m pip install -r requirements/grpo.txt
```

修改 `configs/pipeline.linux.json` 后，先生成命令：

```bash
python -m cail --config configs/pipeline.linux.json grpo --round 1
```

执行两轮训练：

```bash
bash scripts/train_grpo.sh 1
bash scripts/train_grpo.sh 2
```

每个 prompt 采样 8 个回复，使用 `group_norm` advantage、KL loss 和规则奖励。第二轮从第一轮保存的模型继续训练。

```bash
bash scripts/train_grpo.sh 1 --resume
```

每轮输出到 `checkpoints/grpo/round1`、`checkpoints/grpo/round2`，`launch.json` 保存启动参数。

## 推理与评测

基础模型、SFT、GRPO 分别推理：

```bash
python -m cail infer --output runs/base/predictions.jsonl
python -m cail infer --adapter checkpoints/sft-adapter --output runs/sft/predictions.jsonl
python -m cail infer --model checkpoints/grpo/round1 --output runs/grpo1/predictions.jsonl
python -m cail infer --model checkpoints/grpo/round2 --output runs/grpo2/predictions.jsonl
```

API 推理在 `inference.base_url` 配置服务地址，`--model` 指定服务中的模型名称：

```bash
python -m cail infer --backend api --model legal-qwen --output runs/api/predictions.jsonl
```

评测命令：

```bash
python -m cail evaluate --predictions runs/base/predictions.jsonl --output runs/base
python -m cail evaluate --predictions runs/sft/predictions.jsonl --output runs/sft
python -m cail evaluate --predictions runs/grpo1/predictions.jsonl --output runs/grpo1
python -m cail evaluate --predictions runs/grpo2/predictions.jsonl --output runs/grpo2
python -m cail compare --run base=runs/base/metrics.json --run sft=runs/sft/metrics.json --run grpo1=runs/grpo1/metrics.json --run grpo2=runs/grpo2/metrics.json --output runs/comparison.md
```

产物包括 `metrics.json`、`records.jsonl`、`bad_cases.jsonl`。指标按单选/多选、学科、题目类型分组。错误分为漏选、多选、部分重叠、全错和格式错误。

官方测试题使用 `--input data/processed/test.jsonl` 推理，再导出答案映射：

```bash
python -m cail infer --input data/processed/test.jsonl --model checkpoints/grpo/round2 --output runs/test/predictions.jsonl
python -m cail submission --predictions runs/test/predictions.jsonl --output runs/test/answers.json
```

## 核心逻辑检查

```bash
python -m unittest discover -s tests -v
```
