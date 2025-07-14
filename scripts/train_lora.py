# LoRA training script placeholder
# scripts/train_lora.py
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- 1. Configuration ---
MODEL_ID = "model/meta-llama/Meta-Llama-3.1-8B-Instruct"   
DATA_PATH = "data/train.json"
OUTPUT_DIR = "model/lora-adapter"

# --- 2. Load Model and Tokenizer ---
#load tokenizer to local
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) # Load tokenizer    #, use_fast=False 
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for causal LM


# # QLoRA configuration
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )


#load the model to local
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    #quantization_config=quant_config,
    torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on modern GPUs
    device_map="auto", # Automatically map model to mulyiple GPU devices
   # load_in_4bit=True # Load model in 4-bit quantization for memory efficiency
)




# --- 3. LoRA Configuration ---
lora_config = LoraConfig(
    r=8, # Rank of the LoRA adapter to control the volume of LoRA layer.  ΔW ≈ A × B A∈Rd×r，B∈Rr×d
    lora_alpha=16, # Scaling factor for LoRA layers, controls the strength of the adapter
    target_modules=["q_proj", "v_proj"], # Target modules for LoRA, typically the query and value projections in attention layers
    lora_dropout=0.05, # Dropout rate for LoRA layers to prevent overfitting
    bias="none",  
    task_type=TaskType.CAUSAL_LM # Task type for the model, here it's a causal language model
)
# applu LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 4. Preprocess Data ---
label_map = {1: "积极", 0: "消极"}
def preprocess(example):
    prompt = (
        f"<s>[INST] 请判断下面商品评论的情感（积极/消极）：评论内容如下：\n"
        f"{example['sentence']} [/INST] {label_map[example['label']]} </s>"
    )
    tokenized = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# load all local JSON file
dataset = load_dataset("json", data_files=DATA_PATH, split="train") 


# all_data = dataset.to_list()
# # only select first 90% of data for training
# train_data = all_data[:90]
# # turn to Dataset object  
# from datasets import Dataset
# train_dataset = Dataset.from_list(train_data)


#data preprocessing 
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names) # remove_columns=train_dataset.column_names to remove original columns ("sentence", "label") and only keep tokenizer columns such as "input_ids",  "labels"


# --- 5. Training ---
trainer = Trainer(
    model=model, # Load the model with LoRA adapter
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR, # Directory to save the model and adapter
        per_device_train_batch_size=4, # Batch size per device (GPU)
        dataloader_num_workers=12, # Number of workers for data loading
        gradient_accumulation_steps=8, # Gradient accumulation steps to simulate larger batch size
        learning_rate=2e-4, # Learning rate for Adam optimization for LoRA adapter parameters training
        num_train_epochs=3, # Number of training epochs 
        fp16=False,
        bf16=True, # Use bf16 for better performance on modern GPUs
        logging_steps=10, # Log every 10 steps
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_steps=100, # Save model every 100 steps
        save_total_limit=2, # Limit the total number of saved checkpoints to 2
    )
    
)

trainer.train(resume_from_checkpoint=False)   # Resume from the last checkpoint if available
 
# --- 6. Save Adapter ---
model.save_pretrained(OUTPUT_DIR) 
tokenizer.save_pretrained(OUTPUT_DIR) 