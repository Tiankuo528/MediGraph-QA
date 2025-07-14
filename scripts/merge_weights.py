# scripts/merge_weights.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_PATH = "model/meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "model/lora-adapter"
MERGED_MODEL_PATH = "model/merged_model"

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Load the LoRA adapter and merge
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

# Save the merged model
print(f"Saving merged model to {MERGED_MODEL_PATH}...")
model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)
print("✅ Merged model saved successfully!")