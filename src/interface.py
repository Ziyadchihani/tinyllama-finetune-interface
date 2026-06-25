import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# إعدادات العمل المحلي
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
LORA_WEIGHTS = "./lora_llm_weights"

print("🔄 Loading Model and Tokenizer for the Interface...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

# دمج أوزان الـ LoRA الخاصة بك
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS, local_files_only=True)
model.to("cuda")
model.eval()

# دالة توليد الردود الخاصة بـ Gradio
def chat_function(message, history):
    # تشكيل الـ Prompt بنفس الأسلوب اللذي تدرب عليه الموديل
    prompt = f"### Human: {message} ### Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # فك تشفير النص واستخراج رد الموديل فقط
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # تنظيف الرد لأخذ ما بعد كلمة Assistant فقط
    if "### Assistant:" in full_response:
        response = full_response.split("### Assistant:")[-1].strip()
    else:
        response = full_response.replace(prompt, "").strip()
        
    return response

# 🚀 بناء واجهة الشات باستخدام Gradio
print("🎉 Launching your Local Chat Interface...")
demo = gr.ChatInterface(
    fn=chat_function,
    title="My Local Fine-Tuned LLM 🤖",
    description="Type your message below to chat with your custom trained TinyLlama model!",
    
)

if __name__ == "__main__":
    demo.launch(share=False)