from pathlib import Path

def chat_text(tokenizer, items, generation=True):
    return tokenizer.apply_chat_template(items, tokenize=False, add_generation_prompt=generation,
                                         enable_thinking=False)


def load_model(path, quantization="none", dtype="auto", device_map="auto", adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    options = {"torch_dtype": dtype if dtype == "auto" else getattr(torch, dtype),
               "device_map": device_map, "trust_remote_code": False}
    if quantization == "4bit":
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_use_double_quant=True,
                                                          bnb_4bit_compute_dtype=options["torch_dtype"] if dtype != "auto" else torch.float16)
    elif quantization != "none":
        raise ValueError(f"未知量化方式: {quantization}")
    model = AutoModelForCausalLM.from_pretrained(path, **options)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer


def merge_adapter(base, adapter, output, dtype="float16"):
    model, tokenizer = load_model(base, dtype=dtype, device_map="cpu", adapter=adapter)
    merged = model.merge_and_unload()
    Path(output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
