import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Force local environment execution to prevent network handshake drops
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Base configuration settings using TinyLlama
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" 
DATASET_NAME = "timdettmers/openassistant-guanaco" 
OUTPUT_DIR = "../lora_llm_weights"

print("🔄 Loading TinyLlama tokenizer and quantized model...")

# 1. Load Tokenizer and configure padding token alignment
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure 4-bit precision quantization parameters optimized for GTX 1650
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Enforce pure Float16 computation
    bnb_4bit_use_double_quant=True,
)

# 3. Load base causal language model structure with Float16 enforcement
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 4. Prepare quantized model layers specifically for K-bit training structures
model = prepare_model_for_kbit_training(model)

# 5. Configure LoRA parameters tailored for the Llama attention architecture
peft_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Specific to Llama family layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 6. Load training dataset (Restricted to 500 samples to prevent VRAM spikes)
dataset = load_dataset(DATASET_NAME, split="train[:500]")

# 7. Set modern SFT training arguments compliant with current library versions
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     
    gradient_accumulation_steps=4,     
    learning_rate=2e-4,
    logging_steps=5,                   # Low logging steps to track performance quickly
    max_steps=30,                      # Short step count to quickly verify loop stability
    fp16=True,                         
    optim="paged_adamw_8bit",          
    dataset_text_field="text",         
    max_length=256                     
)

# 8. Initialize Supervised Fine-Tuning (SFT) Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,        
    args=training_args,                
)

# 🛑 System bypass flags to disable GradScaler and eliminate BFloat16 leakage on GTX 1650
if hasattr(trainer.accelerator, "scaler") and trainer.accelerator.scaler is not None:
    trainer.accelerator.scaler._enabled = False

print("🚀 Initiating TinyLlama QLoRA Fine-Tuning...")
trainer.train()

# 9. Export and save the trained LoRA adapter weights locally
trainer.model.save_pretrained(OUTPUT_DIR)
print(f"🎉 Success! Adapters exported to: {OUTPUT_DIR}")
