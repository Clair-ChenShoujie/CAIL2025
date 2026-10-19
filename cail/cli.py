import argparse
import json
from pathlib import Path

from .io import load_config

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="CAIL 司法推理训练流水线")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pipeline.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="校验、去重并划分数据")
    p.add_argument("--source")
    p.add_argument("--output")
    p = sub.add_parser("ifd", help="计算 IFD")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--model")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("select", help="按 IFD 筛选样本")
    p.add_argument("--input")
    p.add_argument("--scores")
    p.add_argument("--output")
    p = sub.add_parser("cot", help="生成三段式推理链")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("export", help="生成 SFT 和 GRPO 数据")
    p.add_argument("--cot")
    p.add_argument("--train")
    p.add_argument("--dev")
    p.add_argument("--output")
    p = sub.add_parser("sft", help="QLoRA 监督微调")
    p.add_argument("--model")
    p.add_argument("--train")
    p.add_argument("--output")
    p.add_argument("--resume")
    p = sub.add_parser("merge", help="合并 LoRA 权重")
    p.add_argument("--model")
    p.add_argument("--adapter")
    p.add_argument("--output")
    p = sub.add_parser("grpo", help="生成或执行 OpenRLHF GRPO 命令")
    p.add_argument("--model")
    p.add_argument("--prompts")
    p.add_argument("--eval")
    p.add_argument("--output")
    p.add_argument("--round", type=int, choices=(1, 2), default=1)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("infer", help="批量推理")
    p.add_argument("--input")
    p.add_argument("--output", required=True)
    p.add_argument("--model")
    p.add_argument("--adapter")
    p.add_argument("--backend", choices=("local", "api"), default="local")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("evaluate", help="评测并导出错误样本")
    p.add_argument("--gold")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-missing", action="store_true")
    p = sub.add_parser("compare", help="对比多个实验")
    p.add_argument("--run", action="append", required=True, metavar="NAME=METRICS_JSON")
    p.add_argument("--output", required=True)
    p = sub.add_parser("submission", help="导出 id 到答案的 JSON")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = config["paths"]
    processed = Path(paths["processed"])
    model = getattr(args, "model", None) or paths["model"]
    command, result = args.command, None
    if command == "prepare":
        from .data import prepare
        result = prepare(args.source or paths["source"], args.output or processed, **config["data"])
    elif command == "ifd":
        from .ifd import score_file
        score_file(args.input or processed / "train.jsonl", args.output or processed / "ifd_scores.jsonl",
                   model, **config["ifd"], limit=args.limit)
    elif command == "select":
        from .ifd import select_file
        result = select_file(args.input or processed / "train.jsonl", args.scores or processed / "ifd_scores.jsonl",
                             args.output or processed / "selected.jsonl", **config["selection"])
    elif command == "cot":
        from .cot import generate_file
        result = generate_file(args.input or processed / "selected.jsonl", args.output or processed / "cot.jsonl",
                               config["cot"], args.limit)
    elif command == "export":
        from .export import export_data
        result = export_data(args.cot or processed / "cot.jsonl", args.train or processed / "selected.jsonl",
                             args.dev or processed / "dev.jsonl", args.output or processed)
    elif command == "sft":
        from .sft import train
        train(config["sft"], model, args.train or processed / "sft_train.jsonl", args.output or paths["sft_adapter"], args.resume)
    elif command == "merge":
        from .modeling import merge_adapter
        merge_adapter(model, args.adapter or paths["sft_adapter"], args.output or paths["sft_merged"], config["sft"]["dtype"])
    elif command == "grpo":
        from .grpo import launch
        initial = paths["sft_merged"] if args.round == 1 else str(Path(paths["grpo"]) / "round1")
        launch(config["grpo"], args.model or initial, args.prompts or str(processed / "grpo_train.jsonl"),
               args.output or str(Path(paths["grpo"]) / f"round{args.round}"),
               args.eval or str(processed / "grpo_dev.jsonl"), args.execute, args.resume)
    elif command == "infer":
        from .inference import infer_file
        result = infer_file(args.input or processed / "dev.jsonl", args.output, model, config["inference"],
                            args.adapter, args.backend, args.limit)
    elif command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.gold or processed / "dev.jsonl", args.predictions, args.output, args.allow_missing)
    elif command == "compare":
        from .evaluation import compare_runs
        runs = []
        for item in args.run:
            if "=" not in item:
                parser.error("--run 使用 NAME=METRICS_JSON 格式")
            runs.append(item.split("=", 1))
        compare_runs(runs, args.output)
    elif command == "submission":
        from .evaluation import submission
        result = submission(args.predictions, args.output)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
