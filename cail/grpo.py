import os
import shlex
import subprocess
import sys
from pathlib import Path

from .io import write_json


def build_command(config, model, prompts, output, reward_path, eval_path=None, resume=False):
    values = {
        "pretrain": model, "prompt_data": prompts, "input_key": "prompt", "label_key": "label",
        "remote_rm_url": str(Path(reward_path).resolve()), "save_path": output,
        "ckpt_path": str(Path(output) / "checkpoints"),
        "advantage_estimator": "group_norm", "n_samples_per_prompt": 8,
        "actor_num_nodes": 1, "actor_num_gpus_per_node": config.get("actor_gpus", 1),
        "ref_num_nodes": 1, "ref_num_gpus_per_node": config.get("actor_gpus", 1),
        "vllm_num_engines": config.get("vllm_engines", 1),
        "vllm_tensor_parallel_size": config.get("tensor_parallel", 1),
        "vllm_gpu_memory_utilization": config.get("vllm_memory", 0.5),
        "micro_train_batch_size": config.get("micro_train_batch_size", 1),
        "train_batch_size": config.get("train_batch_size", 128),
        "micro_rollout_batch_size": config.get("micro_rollout_batch_size", 8),
        "rollout_batch_size": config.get("rollout_batch_size", 128),
        "max_epochs": config.get("max_epochs", 1), "num_episodes": config.get("num_episodes", 1),
        "prompt_max_len": config.get("prompt_max_len", 2048),
        "generate_max_len": config.get("generate_max_len", 2048),
        "zero_stage": 3, "actor_learning_rate": config.get("learning_rate", 5e-7),
        "init_kl_coef": config.get("kl_coef", 0.01), "kl_estimator": "k3",
        "temperature": config.get("temperature", 1.0), "top_p": 0.95,
        "save_steps": config.get("save_steps", 50), "seed": config.get("seed", 42)}
    if eval_path:
        values.update(eval_dataset=eval_path, eval_steps=config.get("eval_steps", 50), eval_n_samples_per_prompt=1)
    flags = ["bf16", "gradient_checkpointing", "use_kl_loss", "save_hf_ckpt",
             "colocate_all_models", "vllm_enable_sleep", "deepspeed_enable_sleep"]
    if resume:
        flags.append("load_checkpoint")
    command = [sys.executable, "-m", "openrlhf.cli.train_ppo_ray"]
    for key, value in values.items():
        command.extend(["--" + key, str(value)])
    command.extend("--" + flag for flag in flags)
    return command


def launch(config, model, prompts, output, eval_path=None, execute=False, resume=False):
    root = Path(__file__).resolve().parent.parent
    command = build_command(config, model, prompts, output, root / "scripts" / "reward_func.py", eval_path, resume)
    write_json(Path(output) / "launch.json", {"command": command, "config": config, "openrlhf": "0.8.0"})
    print(shlex.join(command))
    if execute:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
        subprocess.run(command, check=True, cwd=root, env=environment)
    return command
