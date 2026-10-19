import os

from .io import read_jsonl, write_json


def pack_training_example(tokenizer, messages, max_length):
    from .modeling import chat_text
    if len(messages) != 3 or [item["role"] for item in messages] != ["system", "user", "assistant"]:
        raise ValueError("SFT 数据需要 system、user、assistant 三条消息")
    prompt = chat_text(tokenizer, messages[:-1])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(messages[-1]["content"] + tokenizer.eos_token, add_special_tokens=False)
    if len(prompt_ids) + len(response_ids) > max_length:
        return None
    return {"input_ids": prompt_ids + response_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(response_ids)),
            "labels": [-100] * len(prompt_ids) + response_ids}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(row["input_ids"]) for row in features)
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in features:
            padding = length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def train(config, model_path, train_path, output_dir, resume=None):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed
    from .modeling import load_model

    set_seed(config.get("seed", 42))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    model, tokenizer = load_model(model_path, quantization=config.get("quantization", "4bit"),
                                  dtype=config.get("dtype", "bfloat16"), device_map={"": rank})
    if config.get("quantization", "4bit") == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()
    lora = LoraConfig(r=config.get("lora_rank", 64), lora_alpha=config.get("lora_alpha", 128),
                      lora_dropout=config.get("lora_dropout", 0.05), bias="none", task_type="CAUSAL_LM",
                      target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    features, skipped = [], []
    for row in read_jsonl(train_path):
        packed = pack_training_example(tokenizer, row["messages"], config.get("max_length", 4096))
        if packed is None:
            skipped.append(row["id"])
        else:
            features.append(packed)
    if not features:
        raise ValueError("没有可用于 SFT 的样本")
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 16),
        learning_rate=config.get("learning_rate", 2e-4), lr_scheduler_type="cosine",
        warmup_ratio=config.get("warmup_ratio", 0.03), weight_decay=0.01,
        max_grad_norm=1.0, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=config.get("dtype", "bfloat16") == "bfloat16", fp16=config.get("dtype") == "float16",
        logging_steps=config.get("logging_steps", 10), save_steps=config.get("save_steps", 100),
        save_total_limit=3, optim="adamw_torch", report_to="none",
        ddp_find_unused_parameters=False, seed=config.get("seed", 42), remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(features),
                      data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if trainer.is_world_process_zero():
        write_json(os.path.join(output_dir, "training_summary.json"), {
            "samples": len(features), "too_long": skipped, "metrics": result.metrics, "config": config})
