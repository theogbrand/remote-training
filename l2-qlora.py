import os

from copy import deepcopy
from random import randrange
from functools import partial

import torch
import accelerate
import bitsandbytes as bnb

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.integrations import WandbCallback
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)
from trl import SFTTrainer

base_model = "meta-llama/Llama-2-7b-chat-hf"
instruction_dataset = "ogbrandt/pjf_llama_instruction_prep"
new_model = "llama-2-7b-chat-pjf"

import wandb

run = wandb.init(
    project="llama-7b-pjf-ft-v2",  # Project name.
    name="log model end",          # name of the run within this project.
    config={                     # Configuration dictionary.
        "split": "train"
    },
    group="fafo",             # Group runs. This run belongs in "dataset".
    tags=["end"],            # Tags. More dynamic, low-level grouping.
    notes=""  # Description about the run.
)  # Check out the other parameters in the `wandb.init`!

os.environ["WANDB_PROJECT"] = "llama-7b-pjf-ft-v1"

# Log model when running HF Trainer reporting to wandb.
os.environ["WANDB_LOG_MODEL"] = "end"  # Apparently this is deprecated in version 5 of transformers.

# Use wandb to watch the gradients & model parameters.
os.environ["WANDB_WATCH"] = "all"

print("wandb log model:", os.environ.get("WANDB_LOG_MODEL"))

dataset = load_dataset(instruction_dataset)
dataset = dataset["train"].train_test_split(test_size=0.1).shuffle(seed=4242)

tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True,add_bos_token=True,)
# https://stackoverflow.com/questions/76446228/setting-padding-token-as-eos-token-when-using-datacollatorforlanguagemodeling-fr
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"":0},  # Auto selects device to put model on.
)
model.config.use_cache = False

import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print(model)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)  # Explicitly specify!

# https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # lm_head is often excluded.
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)
print(modules)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=modules,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# For `PeftModel`s we can use `get_nb_trainable_parameters` to get the param counts.
trainable, total = model.get_nb_trainable_parameters()  # Returns a Tuple.
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

# Set training parameters
training_params = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps=1,
    gradient_checkpointing = True,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type = "cosine",
    report_to="wandb",
)

# https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

max_length = get_max_length(model)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_length,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

results = trainer.train()

# run.finish()

# trainer.model.save_pretrained("l2-hf-chat-ft")
trainer.save_model()
wandb.finish()
# model.config.use_cache = True
# model.eval()

# import wandb

# artifact = wandb.Artifact(name="model_weights", type="model")

# # Recursively add a directory
# artifact.add_dir(local_path="./outputs", name="adapters")
# run = wandb.init(project="llama-7b-pjf-ft-v2", job_type="model")
