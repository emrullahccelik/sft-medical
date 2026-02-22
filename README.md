# 🏥 Medical SFT — Türkçe Tıbbi Dil Modeli Fine-Tuning

Türkçe tıbbi sorulara doğru ve kapsamlı yanıtlar üretebilen dil modelleri oluşturmak için **Supervised Fine-Tuning (SFT)** pipeline'ı.

## 📋 Proje Hakkında

Bu proje, açık kaynaklı LLM'leri (Llama 3.2, Qwen3 vb.) Türkçe tıbbi veri seti üzerinde fine-tune ederek, tıp alanında uzmanlaşmış asistanlar oluşturmayı amaçlar. Eğitim süreci **LoRA/QLoRA** ile parametre verimli şekilde gerçekleştirilir ve düşük VRAM'li GPU'larda (4GB+) çalışabilir.

### ✨ Özellikler

- 🚀 **Unsloth** ile 2x hızlı fine-tuning ve %60+ VRAM tasarrufu
- 🧬 **QLoRA** — 4-bit quantization ile düşük GPU bellek kullanımı
- 📊 **Otomatik metrik hesaplama** — BLEU, ROUGE, BERTScore
- 📦 **Tek komutla HF Hub'a yükleme** — Model kartı otomatik oluşturulur
- 🇹🇷 **Türkçe tıbbi veri seti** — 47K+ soru-cevap çifti

---

## 🤗 Yayınlanan Model

<a href="https://huggingface.co/emrullahcelik/Llama-3.2-1B-Instruct-Medical-SFT">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Llama--3.2--1B--Medical--SFT-blue?style=for-the-badge" alt="Hugging Face Model"/>
</a>

**[emrullahcelik/Llama-3.2-1B-Instruct-Medical-SFT](https://huggingface.co/emrullahcelik/Llama-3.2-1B-Instruct-Medical-SFT)** — Bu pipeline ile eğitilmiş ve HF Hub'a yüklenmiş model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "emrullahcelik/Llama-3.2-1B-Instruct-Medical-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın."},
    {"role": "user", "content": "Hipertansiyon nedir ve tedavisi nasıl yapılır?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, top_p=0.9)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
```

## 🗂️ Proje Yapısı

```
qwen-sft-medical/
├── app/
│   ├── .env                     # HF_TOKEN, HF_USERNAME
│   ├── config.py                # Proje ayarları ve dizin yolları
│   ├── core/
│   │   └── logger.py            # Loguru tabanlı loglama
│   ├── unsloth-sft/             # 🔥 Unsloth ile SFT pipeline
│   │   ├── train.py             # Model eğitimi
│   │   ├── inference.py         # Çıkarım & demo
│   │   ├── test.py              # Metrik değerlendirme
│   │   ├── publish.py           # HF Hub'a yükleme
│   │   └── metrics/             # Test sonuçları
│   │       ├── results.json
│   │       └── scores.png
│   └── models/
│       ├── pre-trained/         # İndirilen base modeller (cache)
│       ├── checkpoints/         # Eğitim sırasındaki checkpointler
│       └── finetuned/           # Son fine-tuned model & adapter
├── .gitignore
└── README.md
```

---

## 🚀 Hızlı Başlangıç

### 1. Ortamı Hazırla

```bash
# Virtual environment oluştur
python3 -m venv .venv
source .venv/bin/activate

# PyTorch (CUDA 12.4+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Bağımlılıklar
pip install unsloth transformers trl datasets loguru python-dotenv tensorboard
pip install evaluate nltk matplotlib rouge-score bert-score  # Test için
```

### 2. `.env` Dosyasını Ayarla

```bash
# app/.env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HF_USERNAME=kullanici_adi
```

### 3. Eğitimi Başlat

```bash
python -m app.unsloth-sft.train
```

### 4. Modeli Test Et

```bash
# Hızlı çıkarım (3 örnek soru)
python -m app.unsloth-sft.inference

# Metrik değerlendirme (BLEU, ROUGE, BERTScore)
python -m app.unsloth-sft.test
```

### 5. HF Hub'a Yükle

```bash
python -m app.unsloth-sft.publish
```

---

## 🧬 Eğitim Detayları

### Model & Veri Seti

| Parametre | Değer |
|-----------|-------|
| **Base Model** | [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) |
| **Veri Seti** | [turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1](https://huggingface.co/datasets/turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1) |
| **Veri Boyutu** | 47,169 train / 4,148 val / 4,148 test |
| **Chat Template** | Llama 3.1 |

### LoRA Konfigürasyonu

| Parametre | Değer |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable Parameters | 11.27M / 1.25B (%0.90) |

### Eğitim Hiperparametreleri

| Parametre | Değer |
|-----------|-------|
| Epochs | 1 |
| Batch Size | 2 |
| Gradient Accumulation | 4 (effective: 8) |
| Learning Rate | 2e-4 |
| LR Scheduler | Linear |
| Optimizer | AdamW 8-bit |
| Precision | BFloat16 |
| Quantization | 4-bit (QLoRA) |
| Max Seq Length | 1024 |
| Packing | ✅ (5x hızlandırma) |
| Gradient Checkpointing | Unsloth optimized |
| Train Strategy | `train_on_responses_only` |

### Eğitim Sonuçları

| Metrik | Değer |
|--------|-------|
| **Başlangıç Loss** | 3.14 |
| **Son Loss** | ~1.57 |
| **Loss Azalma** | ~%50 |
| **Eğitim Süresi** | ~6 saat |
| **GPU** | NVIDIA RTX 3050 Ti (4GB) |
| **Peak VRAM** | 2.88 GB (%72) |

---

## 📊 Değerlendirme

Test scripti (`test.py`) aşağıdaki metrikleri hesaplar:

- **BLEU** — N-gram örtüşme skoru
- **ROUGE-1 / ROUGE-2 / ROUGE-L** — Recall-oriented metin benzerliği
- **BERTScore** — Anlamsal benzerlik (precision, recall, F1)

Sonuçlar `app/unsloth-sft/metrics/` altında JSON ve grafik olarak kaydedilir.

---

## 🛠️ Komut Referansı

| Komut | Açıklama |
|-------|----------|
| `python -m app.unsloth-sft.train` | Modeli eğit |
| `python -m app.unsloth-sft.inference` | 3 örnek tıbbi soruyla test et |
| `python -m app.unsloth-sft.test` | BLEU/ROUGE/BERTScore hesapla |
| `python -m app.unsloth-sft.publish` | HF Hub'a merge & yükle |

---

## 📝 Sistem Gereksinimleri

- **GPU:** NVIDIA GPU (4GB+ VRAM), CUDA 8.6+
- **Python:** 3.10+
- **OS:** Linux (WSL2 desteklenir)
- **Disk:** ~10GB (model cache + checkpoints)

---

## ⚠️ Sorumluluk Reddi

Bu proje yalnızca **araştırma ve eğitim** amaçlıdır. Üretilen modeller **tıbbi teşhis veya tedavi için kullanılmamalıdır**. Sağlık sorunlarınız için mutlaka bir sağlık profesyoneline danışın.

---

## 📄 Lisans

- **Model:** Llama 3.2 Community License
- **Veri Seti:** Veri seti lisansına tabidir

---

<p align="center">
  <i>Unsloth 🦥 ile hızlandırılmış fine-tuning</i>
</p>