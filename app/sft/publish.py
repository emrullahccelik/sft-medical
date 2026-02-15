import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, ModelCard, ModelCardData
from app.config import CACHE_DIR, SAVE_DIR, HF_TOKEN, HF_USERNAME

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1"

# Yeni modelin adı
NEW_MODEL_NAME = "Qwen3-0.6B-Medical-SFT"

REPO_ID = f"{HF_USERNAME}/{NEW_MODEL_NAME}"
# ---------------------------------------------------------

MODEL_CARD_TEMPLATE = """---
language:
  - tr
license: apache-2.0
library_name: transformers
tags:
  - medical
  - turkish
  - sft
  - fine-tuned
  - qwen3
  - lora
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
| **Yöntem** | SFT (Supervised Fine-Tuning) + LoRA |
| **Dil** | Türkçe 🇹🇷 |
| **Veri Seti** | [{dataset}](https://huggingface.co/datasets/{dataset}) |
| **Lisans** | Apache 2.0 |

## 🧬 Eğitim Bilgileri

- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.05
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision:** bfloat16
- **Optimizer:** AdamW

## 💡 Kullanım

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {{"role": "system", "content": "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın."}},
    {{"role": "user", "content": "Hamilelikte baş ağrısı için hangi ilaçlar güvenlidir?"}}
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
        base_model=BASE_MODEL_ID,
        dataset=DATASET_ID,
        model_name=NEW_MODEL_NAME,
        repo_id=REPO_ID,
    )


def main():
    if not HF_TOKEN:
        print("❌ HATA: HF_TOKEN bulunamadı! Lütfen .env dosyanızı kontrol edin.")
        return

    print(f"🔄 Base model yükleniyor: {BASE_MODEL_ID}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
            cache_dir=CACHE_DIR
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, 
            trust_remote_code=False,
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return

    adapter_path = f"{SAVE_DIR}{BASE_MODEL_ID}/"
    print(f"🔄 Adapter yükleniyor: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"❌ Adapter yükleme hatası: {e}")
        return
    
    print("🔄 Model birleştiriliyor (Merge & Unload)...")
    model = model.merge_and_unload()

    print(f"🚀 Hugging Face Hub'a yükleniyor: {REPO_ID}")
    try:
        # Modeli yükle
        model.push_to_hub(REPO_ID, token=HF_TOKEN)
        # Tokenizer'ı yükle
        tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)

        # Model kartını yükle
        print("📝 Model kartı oluşturuluyor...")
        api = HfApi(token=HF_TOKEN)
        model_card_content = create_model_card()
        api.upload_file(
            path_or_fileobj=model_card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
        )
        
        print(f"\n✅ İŞLEM BAŞARILI!")
        print(f"🔗 Model Linki: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Yükleme sırasında hata oluştu: {e}")

if __name__ == "__main__":
    main()