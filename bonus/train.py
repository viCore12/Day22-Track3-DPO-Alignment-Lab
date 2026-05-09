"""
DPO Training script for Customer-service Chatbot (Bonus Challenge)
Re-uses Unsloth library similar to main Lab 22.
"""
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from trl import DPOTrainer
import pandas as pd
from datasets import Dataset

PatchDPOTrainer()

max_seq_length = 512
output_dir = "adapters/dpo-bonus"

print("1. Loading Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-3B-bnb-4bit", # Can load SFT adapter here instead
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("2. Loading Data...")
# Đọc file parquet sinh ra từ generate_data.py
df = pd.read_parquet("data/pairs.parquet")
dataset = Dataset.from_pandas(df)

# Hàm chuẩn hóa format cho TRL DPOTrainer
def format_dpo_data(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"] + tokenizer.eos_token,
        "rejected": example["rejected"] + tokenizer.eos_token,
    }

dataset = dataset.map(format_dpo_data)

print("3. Training DPO...")
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None, # Unsloth implicitly handles reference model for LoRA
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 5e-7,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = output_dir,
    ),
    beta = 0.1,
    train_dataset = dataset,
    tokenizer = tokenizer,
    max_length = max_seq_length,
    max_prompt_length = 256,
)

dpo_trainer.train()

print(f"4. Saving to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
