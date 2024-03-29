import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset
import wandb

run = wandb.init(
    project="mistral-pjf-v3",  # Project name.
    name="init",          # name of the run within this project.
    job_type="training",
    group="init",             # Group runs. This run belongs in "dataset".
    notes="adapters only"  # Description about the run.
)  # Check out the other parameters in the `wandb.init`!

os.environ["WANDB_PROJECT"] = "mistral-pjf-v3"

# Log model when running HF Trainer reporting to wandb.
os.environ["WANDB_LOG_MODEL"] = "end"  # Apparently this is deprecated in version 5 of transformers.

# Use wandb to watch the gradients & model parameters.
os.environ["WANDB_WATCH"] = "all"

print("wandb log model:", os.environ.get("WANDB_LOG_MODEL"))

accelerator = Accelerator()

hf_model_path="mistralai/Mistral-7B-v0.1"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    hf_model_path,    
    # device_map={"": accelerator.process_index},
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=False)    # fast tokenizer sometimes ignores the added tokens

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token, 
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
tokenizer.padding_side = "left"
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id

# Add adapters to model
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'o_proj'],
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.config.use_cache = False

import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print("model", model)

# Load dataset
dataset = load_dataset("ogbrandt/pjf_chatml_prep")
dataset = dataset["train"].train_test_split(test_size=0.1).shuffle(seed=4242)

# Tokenize dataset
def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )

dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    # remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)

# For `PeftModel`s we can use `get_nb_trainable_parameters` to get the param counts.
trainable, total = model.get_nb_trainable_parameters()  # Returns a Tuple.
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokenlist=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokenlist])

    input_ids,labels,attention_masks = [],[],[]
    for tokens in tokenlist:
        pad_len=tokens_maxlen-len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0
        input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
        labels.append( tokens + [-100]*pad_len )    
        attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

    batch={
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks)
    }
    return batch

bs=1        # batch size
ga_steps=1  # gradient acc. steps
epochs=1
steps_per_epoch=len(dataset_tokenized["train"])//(accelerator.state.num_processes*bs*ga_steps)

args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_8bit",
    learning_rate=0.0002,
    group_by_length=True,
    bf16=True,
    ddp_find_unused_parameters=False,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    max_seq_length=512,
    packing=False,
    dataset_text_field="text",
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False

results = trainer.train()

trainer.save_model()

run.finish()

model.config.use_cache = True
model.eval()