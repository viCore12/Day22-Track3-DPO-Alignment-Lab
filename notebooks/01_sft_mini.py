# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB1 — SFT-mini: Build the Lab 21 SFT checkpoint inline
#
# **Stack:** Unsloth + LoRA r=16 + bitsandbytes 4-bit base + 1k VN Alpaca, 1 epoch.
# Maps to deck §1 (why SFT alone insufficient — motivates the upcoming DPO step) +
# deck §3 (DPO will need this SFT checkpoint as initial policy).
#
# > **Mục tiêu:** tạo 1 SFT adapter "đủ tốt" để DPO có gì align lên. Đây là
# > Lab 21 thu gọn — nếu bạn đã hoàn thành Lab 21 sibling repo
# > ([VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA)),
# > bạn có thể SKIP notebook này và copy adapter cũ vào `adapters/sft-mini/`.
# >
# > Nếu chưa, notebook này build từ đầu trong ~10 phút trên T4 (15 phút trên Colab CPU runtime — *đừng làm vậy*).

# %% [markdown]
# ## 0. Setup

# %%
import os
from pathlib import Path

# Tier detection. Defaults to T4 if env not set.
COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
assert COMPUTE_TIER in ("T4", "BIGGPU"), f"Invalid COMPUTE_TIER: {COMPUTE_TIER}"

# Tier-specific hyperparameters
if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    MAX_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:  # BIGGPU
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    MAX_LEN = 1024
    PER_DEVICE_BATCH = 2
    GRAD_ACCUM = 4

SFT_DATASET = os.environ.get("SFT_DATASET", "")
SFT_SLICE = 1000
NUM_EPOCHS = 1

# Dataset priority list (tried in order until one succeeds)
# Primary: 5CD-AI/Vietnamese-alpaca-cleaned (may be gated — needs HF login)
# Fallback 1: bkai-foundation-models/vi-alpaca (public)
# Fallback 2: 5CD-AI/Vietnamese-alpaca-gpt4-gg-translated (public)
SFT_DATASET_CANDIDATES = [
    SFT_DATASET,  # user override (empty string = skip)
    "5CD-AI/Vietnamese-alpaca-cleaned",
    "bkai-foundation-models/vi-alpaca",
    "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated",
]
SFT_DATASET_CANDIDATES = [d for d in SFT_DATASET_CANDIDATES if d]  # remove empty

def get_repo_root():
    """Robustly find the repo root by looking for 'notebooks' or 'adapters' folder."""
    curr = Path.cwd().resolve()
    # Check current, then 2 levels up (for Colab/Notebooks context)
    for _ in range(3):
        if (curr / "notebooks").exists() or (curr / "adapters").exists():
            return curr
        curr = curr.parent
    return Path.cwd().resolve()

REPO_ROOT = get_repo_root()
ADAPTER_OUT = REPO_ROOT / "adapters" / "sft-mini"
ADAPTER_OUT.mkdir(parents=True, exist_ok=True)

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"SFT_DATASET:     {SFT_DATASET}  (slice: {SFT_SLICE})")
print(f"max_seq_length:  {MAX_LEN}")
print(f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"output:          {ADAPTER_OUT}")

# %%
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)")

# %% [markdown]
# ## 1. Load base model with Unsloth
#
# Unsloth bundles patched 4-bit kernels — that's how Qwen2.5-3B (or 7B) stays
# in T4 / A100 budget. The `FastLanguageModel.from_pretrained` call returns a
# 4-bit quantized base; `get_peft_model` attaches the LoRA adapter on top.

# %%
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,                # auto: bf16 on Ampere+, fp16 on Turing
    load_in_4bit=True,
)

# Critical for batch training — Qwen tokenizers ship without pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer.pad_token = eos_token")

# Fix: set ChatML template if tokenizer doesn't have one (can happen with some
# Unsloth builds or when loading from a cached version without template metadata).
if tokenizer.chat_template is None:
    CHATML_TEMPLATE = (
        "{%- if tools %}"
        "    {{- '<|im_start|>system\n' }}"
        "    {%- if messages[0]['role'] == 'system' %}"
        "        {{- messages[0]['content'] }}"
        "    {%- else %}"
        "        {{- 'You are a helpful assistant.' }}"
        "    {%- endif %}"
        "    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}"
        "    {%- for tool in tools %}"
        "        {{- \"\\n\" }}"
        "        {{- tool | tojson }}"
        "    {%- endfor %}"
        "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}"
        "{%- else %}"
        "    {%- if messages[0]['role'] == 'system' %}"
        "        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
        "    {%- else %}"
        "        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
        "    {%- endif %}"
        "{%- endif %}"
        "{%- for message in messages %}"
        "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}"
        "        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}"
        "    {%- elif message.role == \"assistant\" %}"
        "        {{- '<|im_start|>' + message.role }}"
        "        {%- if message.content %}"
        "            {{- '\n' + message.content }}"
        "        {%- endif %}"
        "        {%- for tool_call in message.tool_calls %}"
        "            {%- if tool_call.function is defined %}"
        "                {%- set tool_call = tool_call.function %}"
        "            {%- endif %}"
        "            {{- '\n<tool_call>\n{\"name\": \"' }}"
        "            {{- tool_call.name }}"
        "            {{- '\", \"arguments\": ' }}"
        "            {{- tool_call.arguments | tojson }}"
        "            {{- '}\n</tool_call>' }}"
        "        {%- endfor %}"
        "        {{- '<|im_end|>\n' }}"
        "    {%- elif message.role == \"tool\" %}"
        "        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}"
        "            {{- '<|im_start|>user' }}"
        "        {%- endif %}"
        "        {{- '\n<tool_response>\n' }}"
        "        {{- message.content }}"
        "        {{- '\n</tool_response>' }}"
        "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}"
        "            {{- '<|im_end|>\n' }}"
        "        {%- endif %}"
        "    {%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "    {{- '<|im_start|>assistant\n' }}"
        "{%- endif %}"
    )
    tokenizer.chat_template = CHATML_TEMPLATE
    print("Set tokenizer.chat_template = Qwen2.5 ChatML with tool support (was missing)")
else:
    print(f"tokenizer.chat_template already set ({len(tokenizer.chat_template)} chars)")

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,           # Unsloth supports dropout=0 for free speed
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # 30% VRAM savings
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 2. Load + format VN Alpaca slice
#
# Try candidate datasets in order until one loads successfully.
# If 5CD-AI/Vietnamese-alpaca-cleaned is gated, auto-falls back to
# bkai-foundation-models/vi-alpaca (public, same Alpaca format).

# %%
from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN")  # needed only for gated datasets

ds = None
SFT_DATASET_USED = None
for candidate in SFT_DATASET_CANDIDATES:
    try:
        print(f"Trying dataset: {candidate} ...")
        ds = load_dataset(
            candidate,
            split=f"train[:{SFT_SLICE}]",
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        SFT_DATASET_USED = candidate
        print(f"✓ Loaded {len(ds)} rows from '{candidate}'")
        break
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        continue

assert ds is not None, (
    "All dataset candidates failed. "
    "Set HF_TOKEN env var to access gated datasets, or check your internet connection."
)
print(f"\nUsing dataset: {SFT_DATASET_USED}")
print(f"Columns: {ds.column_names}")
print(f"\nFirst row:\n{ds[0]}")

# %%
# Alpaca → ChatML format (Qwen2.5's native template)
def format_alpaca_to_chat(row):
    messages = []
    if row.get("instruction"):
        prompt = row["instruction"]
        if row.get("input"):
            prompt += "\n\n" + row["input"]
        messages.append({"role": "user", "content": prompt})
    if row.get("output"):
        messages.append({"role": "assistant", "content": row["output"]})
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback: plain ChatML string (never fails)
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        text = "\n".join(parts) + "\n"
    return {"text": text}


ds_formatted = ds.map(format_alpaca_to_chat, remove_columns=ds.column_names)
print(f"\nSample formatted text (first 500 chars):\n{ds_formatted[0]['text'][:500]}")

# %% [markdown]
# ## 3. Train SFT-mini

# %%
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir=str(ADAPTER_OUT.parent / "sft-mini-checkpoints"),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",        # Save only at the end via trainer.model.save_pretrained
    optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=42,
    max_seq_length=MAX_LEN,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=ds_formatted,
)

# %%
train_result = trainer.train()
print(f"\nFinal train loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ### 3a. Plot loss curve (deliverable: `02_sft_loss.png`)

# %%
import matplotlib.pyplot as plt

losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
steps = [log["step"] for log in trainer.state.log_history if "loss" in log]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.2)
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title(f"SFT-mini loss · {COMPUTE_TIER} · {BASE_MODEL.split('/')[-1]} · {SFT_SLICE} samples")
ax.grid(True, alpha=0.3)
fig.tight_layout()
screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "02-sft-loss.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Save adapter + sanity-check generation

# %%
trainer.model.save_pretrained(str(ADAPTER_OUT))
tokenizer.save_pretrained(str(ADAPTER_OUT))
print(f"Saved SFT adapter to {ADAPTER_OUT}")

# %%
# Sanity: generate 1 sample to confirm the adapter loaded correctly.
FastLanguageModel.for_inference(model)
prompt = "Giải thích ngắn gọn (3-4 câu) thuật toán quicksort hoạt động thế nào."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda")
with torch.no_grad():
    out = model.generate(input_ids=inputs, max_new_tokens=200, do_sample=False)
generated = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"PROMPT: {prompt}\n")
print(f"SFT-mini response:\n{generated}")

# %% [markdown]
# ## 5. Vibe-coding callout
#
# Bạn vừa tái tạo Lab 21 trong ~10 phút. Một câu hỏi để brainstorm:
#
# > **Thật ra, bạn cần *bao nhiêu* SFT để DPO có ý nghĩa?**
# >
# > Thử thay `SFT_SLICE = 1000` → `100`. Re-run NB1 → NB3 với SFT yếu hơn.
# > Quan sát: reward gap có còn tăng được không? Output coherent không?
# >
# > Đây là 1 design decision *think-hard zone* (xem VIBE-CODING.md): không có
# > đáp án sẵn trong deck. Hypothesize trước, train sau, viết kết quả vào
# > `submission/REFLECTION.md` § 6.
#
# **Next:** NB2 — load + format preference data.
