import os
from pathlib import Path
from dotenv import load_dotenv

# Projenin kök dizini (config.py'nin bulunduğu yer)
BASE_DIR = Path(__file__).resolve().parent

# .env dosyasının yolu (app/.env)
ENV_PATH = BASE_DIR / ".env"

# .env dosyasını yükle
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f"Uyarı: .env dosyası bulunamadı: {ENV_PATH}")

# Ayarlar
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------
# Dizin Yolları
# ---------------------------------------------------------
CACHE_DIR = f"{BASE_DIR}/models/pre-trained/"
OUTPUT_DIR = f"{BASE_DIR}/models/checkpoints/"
SAVE_DIR = f"{BASE_DIR}/models/finetuned/"
METRICS_DIR = f"{BASE_DIR}/sft/metrics"

# Dizinleri oluştur
for _dir in [CACHE_DIR, OUTPUT_DIR, SAVE_DIR, METRICS_DIR]:
    os.makedirs(_dir, exist_ok=True)
