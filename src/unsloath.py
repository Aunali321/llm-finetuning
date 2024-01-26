import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from peft import LoraConfig
from modal import Stub, Image, Volume, Secret
import os

APP_NAME = "mistral-claude"

stub = Stub(APP_NAME, secrets=[Secret.from_name("huggingface")])

# Volumes for pre-trained models and training runs.
pretrained_volume = Volume.persisted("example-pretrained-vol")
runs_volume = Volume.persisted("example-runs-vol")
VOLUME_CONFIG: dict[str | os.PathLike, Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}

N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_MEM = int(os.environ.get("GPU_MEM", 40))
GPU_CONFIG = modal.gpu.A100(count=N_GPUS, memory=GPU_MEM)


axolotl_image = (
    Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    .pip_install("huggingface_hub==0.19.4", "hf-transfer==0.1.4")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)


max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number

# Load model
model = "unsloth/mistral-7b"

dataset = load_dataset("json", data_files="my_data.jsonl", split="train")

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", 
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

model, tokenizer = setup_chat_format(model, tokenizer)

from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    peft_config=peft_config,
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model()