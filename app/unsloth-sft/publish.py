import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
from app.config import CACHE_DIR, SAVE_DIR, HF_TOKEN, HF_USERNAME
from app.core.logger import logger

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
ORIGINAL_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # HF kartı için orijinal model
DATASET_ID = "turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1"

# Yeni modelin adı
NEW_MODEL_NAME = "Llama-3.2-1B-Instruct-Medical-SFT"

REPO_ID = f"{HF_USERNAME}/{NEW_MODEL_NAME}"
# ---------------------------------------------------------

MODEL_CARD_TEMPLATE = """---
language:
  - tr
license: llama3.2
library_name: transformers
tags:
  - medical
  - turkish
  - sft
  - fine-tuned
  - llama
  - llama-3.2
  - lora
  - unsloth
base_model: {base_model}
datasets:
  - {dataset}
pipeline_tag: text-generation
---

# 🏥 {model_name}

**{model_name}**, Türkçe tıbbi sorulara doğru ve kapsamlı yanıtlar üretmek amacıyla fine-tune edilmiş bir dil modelidir.

## 📋 Model Detayları

| Özellik | Değer |
|---|---|
| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Yöntem** | SFT (Supervised Fine-Tuning) + LoRA (Unsloth) |
| **Dil** | Türkçe 🇹🇷 |
| **Veri Seti** | [{dataset}](https://huggingface.co/datasets/{dataset}) |
| **Lisans** | Llama 3.2 Community License |

## 🧬 Eğitim Bilgileri

- **Framework:** Unsloth + TRL SFTTrainer
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision:** 4-bit quantization (QLoRA)
- **Optimizer:** AdamW 8-bit
- **Learning Rate:** 2e-4
- **Batch Size:** 2 (gradient accumulation: 4, effective: 8)
- **Epochs:** 1
- **Max Sequence Length:** 1024
- **Gradient Checkpointing:** Unsloth optimized
- **Training:** train_on_responses_only

## 📊 Eğitim Sonuçları

- **Train Loss:** 3.14 → 1.57 (~%50 azalma)
- **Eğitim Süresi:** ~6 saat (RTX 3050 Ti 4GB)
- **Peak VRAM:** 2.88 GB (%72)

## 💡 Kullanım

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {{"role": "system", "content": "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın."}},
    {{"role": "user", "content": "Hipertansiyon nedir ve tedavisi nasıl yapılır?"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.8, top_k=20)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
```

## ⚠️ Sorumluluk Reddi

Bu model yalnızca araştırma ve eğitim amaçlıdır. **Tıbbi teşhis veya tedavi için kullanılmamalıdır.** Sağlık sorunlarınız için mutlaka bir sağlık profesyoneline danışın.
"""


def create_model_card():
    """Profesyonel model kartı oluştur"""
    return MODEL_CARD_TEMPLATE.format(
        base_model=ORIGINAL_MODEL_ID,
        dataset=DATASET_ID,
        model_name=NEW_MODEL_NAME,
        repo_id=REPO_ID,
    )


def main():
    if not HF_TOKEN:
        logger.error("❌ HF_TOKEN bulunamadı! Lütfen app/.env dosyanızı kontrol edin.")
        return

    # Base modeli yükle (unsloth versiyonu cache'de mevcut)
    logger.info(f"🔄 Base model yükleniyor: {BASE_MODEL_ID}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
            cache_dir=CACHE_DIR,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=False,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {e}")
        return

    # LoRA adapter'ı yükle
    adapter_path = f"{SAVE_DIR}{BASE_MODEL_ID}/"
    logger.info(f"🔄 Adapter yükleniyor: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        logger.error(f"❌ Adapter yükleme hatası: {e}")
        return

    # Merge & Unload
    logger.info("🔄 Model birleştiriliyor (Merge & Unload)...")
    model = model.merge_and_unload()

    # Hub'a yükle
    logger.info(f"🚀 Hugging Face Hub'a yükleniyor: {REPO_ID}")
    try:
        model.push_to_hub(REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)

        # Model kartını yükle
        logger.info("📝 Model kartı oluşturuluyor...")
        api = HfApi(token=HF_TOKEN)
        model_card_content = create_model_card()
        api.upload_file(
            path_or_fileobj=model_card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
        )

        logger.info(f"✅ İŞLEM BAŞARILI!")
        print(f"🔗 Model Linki: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        logger.error(f"❌ Yükleme sırasında hata oluştu: {e}")


if __name__ == "__main__":
    main()
