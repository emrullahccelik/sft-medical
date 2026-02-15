import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from app.config import CACHE_DIR, SAVE_DIR, HF_TOKEN, HF_USERNAME

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"

# Yeni modelin adı
NEW_MODEL_NAME = "Qwen3-0.6B-Medical-SFT"

REPO_ID = f"{HF_USERNAME}/{NEW_MODEL_NAME}"
# ---------------------------------------------------------

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
        model.push_to_hub(REPO_ID, token=HF_TOKEN, safe_serialization=True)
        # Tokenizer'ı yükle
        tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
        
        print(f"\n✅ İŞLEM BAŞARILI!")
        print(f"🔗 Model Linki: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Yükleme sırasında hata oluştu: {e}")

if __name__ == "__main__":
    main()