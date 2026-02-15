import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os
from app.core.logger import logger
from app.config import CACHE_DIR, OUTPUT_DIR, SAVE_DIR, METRICS_DIR

device = None
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
logger.info(f"Device: {device}")


model_id = "Qwen/Qwen3-0.6B"
MAX_SEQ_LENGTH = 512
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=False,
    cache_dir = CACHE_DIR
)

dataset = load_dataset("turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1")
logger.info(f"Dataset: {dataset}")
SYSTEM_PROMPT = "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın. Soruları doğru, kapsamlı ve anlaşılır biçimde yanıtla."

def format_chat(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return example

logger.info("Veri seti formatlanıyor...")
dataset = dataset.map(format_chat)

def filter_long_examples(example):
    tokens = tokenizer.encode(example["text"], add_special_tokens=False)
    return len(tokens) <= MAX_SEQ_LENGTH

logger.info(f"Uzun veriler filtreleniyor (Max: {MAX_SEQ_LENGTH})...")
initial_count = {k: len(dataset[k]) for k in dataset.keys()}
dataset = dataset.filter(filter_long_examples)
final_count = {k: len(dataset[k]) for k in dataset.keys()}

for split in dataset.keys():
    removed = initial_count[split] - final_count[split]
    if removed > 0:
        logger.warning(f"{split} setinden {removed} adet çok uzun veri çıkarıldı.")
    else:
        logger.success(f"{split} setindeki tüm veriler uzunluk sınırına uygun.")

logger.success("Veri seti hazırlığı tamamlandı.")



lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


num_train_epochs = 1
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
logging_steps = 100
save_strategy = "steps"
save_steps = 600
eval_strategy = "steps"
eval_steps = 600

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}{model_id}/",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_ratio=0.05,
    logging_steps=logging_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    eval_strategy=eval_strategy, 
    eval_steps=eval_steps,
    load_best_model_at_end=True,        # En iyi modeli eğitim sonunda yükle
    metric_for_best_model="loss",        # Loss'a göre karşılaştır
    greater_is_better=False,             # Loss söz konusu olduğu için küçük olan daha iyi
    save_total_limit=20,                  # Sadece son 20 checkpoint'i tut (disk dolmasın diye)
    bf16=True,
    optim="adamw_torch",
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
)

logger.info("Eğitim başlıyor...")
train_result = trainer.train()
logger.success("Eğitim tamamlandı.")

save_path = f"{SAVE_DIR}{model_id}/"
logger.info(f"Model kaydediliyor: {save_path}")
trainer.save_model(save_path)
logger.success("Model başarıyla kaydedildi.")

# Eğitim metriklerini kaydet
import json
import matplotlib.pyplot as plt



# Training metrics'i al
train_metrics = train_result.metrics
train_metrics["train_samples"] = len(dataset["train"])

# Evaluation metrics'i al (eğer varsa)
eval_metrics = trainer.evaluate()

# Tüm metrikleri birleştir
all_metrics = {
    "training": train_metrics,
    "evaluation": eval_metrics
}

# JSON olarak kaydet
metrics_path = os.path.join(METRICS_DIR, "training_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=4, ensure_ascii=False)
logger.success(f"Metrikler kaydedildi: {metrics_path}")

# Training history'yi görselleştir (eğer log history varsa)
if hasattr(trainer.state, 'log_history') and len(trainer.state.log_history) > 0:
    log_history = trainer.state.log_history
    
    # Loss değerlerini çıkar
    train_loss = []
    eval_loss = []
    steps = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            steps.append(entry.get('step', len(train_loss)))
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
    
    # Grafik oluştur
    plt.figure(figsize=(12, 5))
    
    # Training Loss
    plt.subplot(1, 2, 1)
    if train_loss:
        plt.plot(steps[:len(train_loss)], train_loss, 'b-', label='Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Evaluation Loss
    plt.subplot(1, 2, 2)
    if eval_loss:
        eval_steps = [entry.get('step', i) for i, entry in enumerate(log_history) if 'eval_loss' in entry]
        plt.plot(eval_steps, eval_loss, 'r-', label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Validation Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(METRICS_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    logger.success(f"Eğitim grafikleri kaydedildi: {plot_path}")
    plt.close()