from unsloth import FastLanguageModel
import torch
import json
import os
import evaluate
import matplotlib.pyplot as plt
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from app.config import CACHE_DIR, SAVE_DIR, METRICS_DIR
from app.core.logger import logger
from tqdm import tqdm
import nltk

# NLTK verilerini indir (gerekirse)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
DATASET_ID = "turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1"
SYSTEM_PROMPT = "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın. Soruları doğru, kapsamlı ve anlaşılır biçimde yanıtla."
MAX_SEQ_LENGTH = 1024
NUM_TEST_SAMPLES = 500


def load_model():
    """Fine-tuned modeli yükle"""
    adapter_path = f"{SAVE_DIR}{MODEL_ID}/"
    logger.info(f"Model yükleniyor: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        cache_dir=CACHE_DIR,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    """Tek bir prompt için yanıt üret"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def save_results(results):
    """Sonuçları JSON ve grafik olarak kaydet"""
    metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # JSON kaydet
    json_path = os.path.join(metrics_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Sonuçlar kaydedildi: {json_path}")

    # Grafik oluştur
    plt.figure(figsize=(10, 6))
    metrics = list(results.keys())
    values = list(results.values())

    colors = ["#4FC3F7", "#81C784", "#E57373", "#FFB74D", "#CE93D8", "#4DB6AC", "#F06292"]
    plt.bar(metrics, values, color=colors[: len(metrics)])
    plt.title("Llama-3.2-1B-Instruct Medical SFT - Performans Metrikleri")
    plt.ylabel("Skor")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(metrics_dir, "scores.png")
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Grafik kaydedildi: {plot_path}")


def main():
    # Veri setini yükle
    logger.info("Test veri seti yükleniyor...")
    dataset = load_dataset(DATASET_ID, split="test")
    sample_dataset = dataset.select(range(NUM_TEST_SAMPLES))

    model, tokenizer = load_model()

    predictions = []
    references = []

    logger.info(f"Yanıtlar üretiliyor ({NUM_TEST_SAMPLES} örnek)...")
    for item in tqdm(sample_dataset, desc="Test"):
        prompt = item["instruction"]
        reference = item["output"]

        prediction = generate_response(model, tokenizer, prompt)

        predictions.append(prediction)
        references.append(reference)

        # İlk birkaç örneği göster
        if len(predictions) <= 3:
            print(f"\n--- Örnek {len(predictions)} ---")
            print(f"Soru: {prompt}")
            print(f"Beklenen: {reference}")
            print(f"Üretilen: {prediction}")

    # BLEU
    logger.info("BLEU hesaplanıyor...")
    bleu = evaluate.load("bleu")
    bleu_refs = [[ref] for ref in references]
    bleu_results = bleu.compute(predictions=predictions, references=bleu_refs)

    # ROUGE
    logger.info("ROUGE hesaplanıyor...")
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # BERTScore
    logger.info("BERTScore hesaplanıyor...")
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="tr")

    final_results = {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bertscore_precision": sum(bert_results["precision"]) / len(bert_results["precision"]),
        "bertscore_recall": sum(bert_results["recall"]) / len(bert_results["recall"]),
        "bertscore_f1": sum(bert_results["f1"]) / len(bert_results["f1"]),
    }

    print("\n" + "=" * 50)
    print("📊 SONUÇLAR")
    print("=" * 50)
    print(json.dumps(final_results, indent=4))

    save_results(final_results)


if __name__ == "__main__":
    main()
