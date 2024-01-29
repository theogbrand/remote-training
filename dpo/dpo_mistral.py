# -*- coding: utf-8 -*-
# !pip install -q datasets trl peft bitsandbytes sentencepiece wandb

import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb

import wandb
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv() 

# Defined in the secrets tab in Google Colab
hf_token = os.getenv("HF_token")
wb_token = os.getenv("wandb_token")
wandb.login(key=wb_token)
login(token=hf_token)

run = wandb.init(
    project="dpo-35-v0",  # Project name.
    name="log model end",          # name of the run within this project.
    config={                     # Configuration dictionary.
        "split": "train"
    },
    group="fafo",             # Group runs. This run belongs in "dataset".
    tags=["end"],            # Tags. More dynamic, low-level grouping.
    notes=""  # Description about the run.
)  # Check out the

# Log model when running HF Trainer reporting to wandb.
os.environ["WANDB_LOG_MODEL"] = "end"  # Apparently this is deprecated in version 5 of transformers.

# Use wandb to watch the gradients & model parameters.
os.environ["WANDB_WATCH"] = "all"

print("wandb log model:", os.environ.get("WANDB_LOG_MODEL"))

model_name = "ogbrandt/mistral-7b-non-instruction-pjf-ft"
new_model = "mistral-7b-non-instruction-pjf-ft-gpt35-dpo-v0"

# chatml like OG Mistral
tokeniser_model_name = "teknium/OpenHermes-2.5-Mistral-7B"
"""## Format dataset"""

def chatml_format(example):
    # # Format system
    # if len(example['system']) > 0:
    #     message = {"role": "system", "content": example['system']}
    #     system = tokenizer.apply_chat_template([message], tokenize=False)
    # else:
    #     system = ""

    system_prompt = ""

    # Format instruction
    message = {"role": "user", "content": example['prompt']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system_prompt + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Load dataset
dataset = load_dataset("ogbrandt/gpt35_preference_rlaif")['train']

# Save columns
original_columns = dataset.column_names

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Format dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)

# Print sample
dataset[1]

"""## Train model with DPO"""

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model.config.use_cache = False

# Reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    save_strategy="no",
    logging_steps=1,
    output_dir=new_model,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to="wandb",
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model,
    ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

# Fine-tune model with DPO
dpo_trainer.train()

"""## Upload model"""

# Save artifacts
dpo_trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")

# Flush memory
del dpo_trainer, model, ref_model
gc.collect()
torch.cuda.empty_cache()

# Reload model in FP16 (instead of NF4)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, "final_checkpoint")
model = model.merge_and_unload()

# Save model and tokenizer
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# Push them to the HF Hub
model.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
tokenizer.push_to_hub(new_model, use_temp_dir=False, token=hf_token)

"""## Inference"""

# Format prompt
message = [
    {"role": "system", "content": "You are a helpful assistant chatbot."},
    {"role": "user", "content": "What is a Large Language Model?"}
]
tokenizer = AutoTokenizer.from_pretrained(new_model)
prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

# Create pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=new_model,
    tokenizer=tokenizer
)

# Generate text
sequences = pipeline(
    prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1,
    max_length=200,
)
print(sequences[0]['generated_text'])