# 🤖 Local LLM Fine-Tuning Pipeline & Chat UI

A lightweight, hardware-optimized pipeline for fine-tuning a Large Language Model (LLM) locally using **QLoRA (4-bit quantization)** and interacting with it through a web-based chat interface.

This project is designed for consumer-grade GPUs such as the **NVIDIA GTX 1650 (4GB VRAM)** by bypassing common `BFloat16` hardware limitations and reducing memory usage through **Parameter-Efficient Fine-Tuning (PEFT)**.

---

## 📸 Interface Preview

### Chat Interface

![Chat Interface](src/scene/one.png)

### Conversation Example

![Conversation Example](src/scene/two.png)

---

## 🚀 Features

* ✅ Local training and inference
* ✅ Works on low-VRAM GPUs (4GB VRAM)
* ✅ QLoRA fine-tuning with 4-bit NF4 quantization
* ✅ Memory-efficient LoRA adapters
* ✅ Offline execution after model download
* ✅ Interactive Gradio web interface
* ✅ Automatic Float16 compatibility patch for older NVIDIA GPUs

---

## 🛠️ Tech Stack

### Base Model

* TinyLlama-1.1B-intermediate-step-1431k-3T

### Dataset

* timdettmers/openassistant-guanaco

### Libraries

* PyTorch
* Transformers
* PEFT (LoRA)
* TRL (SFTTrainer)
* BitsAndBytes
* Gradio

---

## 📂 Project Structure

```text
.
├── lora_llm_weights/          # Generated after training
├── src/
│   ├── train.py               # QLoRA fine-tuning script
│   ├── interface.py           # Gradio chat application
│   ├── one.png                # Interface screenshot
│   └── two.png                # Conversation screenshot
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies

Install PyTorch with CUDA 11.8 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining packages:

```bash
pip install transformers datasets peft trl bitsandbytes gradio
```

Or install everything from:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Fine-Tuning the Model

Navigate to the source folder:

```bash
cd src
```

Run training:

```bash
python train.py
```

After training completes, the generated LoRA adapters will be saved in:

```text
lora_llm_weights/
```

---

## ⚙️ Float16 Compatibility Patch

Some older NVIDIA GPUs may produce errors similar to:

```text
_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'
```

To ensure compatibility, the training script automatically disables the Hugging Face Accelerator scaler and forces stable Float16 training:

```python
if hasattr(trainer.accelerator, "scaler") and trainer.accelerator.scaler is not None:
    trainer.accelerator.scaler._enabled = False
```

---

## 💬 Running the Chat Interface

Navigate to the source directory:

```bash
cd src
```

Launch the Gradio application:

```bash
python interface.py
```

Open the URL displayed in the terminal:

```text
http://127.0.0.1:7860
```

You can now interact with your fine-tuned TinyLlama model through the web interface.

---

## 🎯 Target Hardware

Tested on:

* NVIDIA GTX 1650 (4GB VRAM)
* CUDA 11.8
* Python 3.10

The project is optimized for low-resource systems using:

* 4-bit quantization
* LoRA adapters
* Gradient accumulation
* Float16 training

---

## 📜 License

This project is intended for educational and research purposes.
For commercial use, please review the licenses of the underlying models and datasets.
