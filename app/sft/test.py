import torch
import json
import os
import evaluate
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from app.config import CACHE_DIR, SAVE_DIR, METRICS_DIR
from tqdm import tqdm
import nltk

# NLTK verilerini indir (gerekirse)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


MODEL_ID = "Qwen/Qwen3-0.6B"

def load_model():
    print("Model ve Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False, cache_dir = CACHE_DIR)
    
    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=False,
        cache_dir = CACHE_DIR
    )
    
    # LoRA adapter
    model = PeftModel.from_pretrained(base_model, f"{SAVE_DIR}{MODEL_ID}/")
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    system_prompt = "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın. Soruları doğru, kapsamlı ve anlaşılır biçimde yanıtla."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Sadece üretilen kısmı al
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def calculate_metrics(predictions, references):
    print("Metrikler hesaplanıyor...")
    
    # BLEU
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)
    
    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=references)
    
    # BERTScore
    bertscore = evaluate.load("bertscore")
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="tr")
    
    results = {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "bertscore_f1": sum(bert_score["f1"]) / len(bert_score["f1"])
    }
    
    return results

def save_results(results):
    # JSON kaydet
    json_path = os.path.join(METRICS_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Sonuçlar kaydedildi: {json_path}")
    
    # Grafik oluştur
    plt.figure(figsize=(10, 6))
    metrics = list(results.keys())
    values = list(results.values())
    
    plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'purple'])
    plt.title("Model Performans Metrikleri")
    plt.ylabel("Skor")
    plt.ylim(0, 1)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
    plot_path = os.path.join(METRICS_DIR, "scores.png")
    plt.savefig(plot_path)
    print(f"Grafik kaydedildi: {plot_path}")

def main():
    # Veri setini yükle (test splitinden küçük bir örneklem)
    print("Veri seti yükleniyor...")
    dataset = load_dataset("turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1", split="test")
    # Hızlı test için 10 örnek
    sample_dataset = dataset.select(range(10)) 
    
    model, tokenizer = load_model()
    
    predictions = []
    references = []
    
    print("Yanıtlar üretiliyor...")
    for item in tqdm(sample_dataset):
        prompt = item["instruction"]
        reference = item["output"]
        
        prediction = generate_response(model, tokenizer, prompt)
        
        predictions.append(prediction)
        references.append([reference]) # BLEU için liste içinde liste formatı gerekebilir ama evaluate.load("bleu") genelde references=[[ref1], [ref2]] ister
    
    # Evaluate references formatını kontrol et
    # BLEU: references list of list of strings
    # ROUGE: references list of strings
    # BERTScore: references list of strings
    
    # Düzenleme: ROUGE ve BERTScore tek referans listesi ister, BLEU çoklu.
    # Evaluate kütüphanesinde BLEU için references formatı: [[ref1_a, ref1_b], [ref2_a, ...]]
    # Bizim veri setinde her promptun tek cevabı var.
    metrics_references = references 
    simple_references = [ref[0] for ref in references] # ROUGE ve BERTScore için
    
    # BLEU
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=predictions, references=metrics_references)
    
    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=simple_references)
    
    # BERTScore
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(predictions=predictions, references=simple_references, lang="tr")
    
    final_results = {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bertscore_precision": sum(bert_results["precision"]) / len(bert_results["precision"]),
        "bertscore_recall": sum(bert_results["recall"]) / len(bert_results["recall"]),
        "bertscore_f1": sum(bert_results["f1"]) / len(bert_results["f1"])
    }
    
    print("\n--- Sonuçlar ---")
    print(json.dumps(final_results, indent=4))
    
    save_results(final_results)

if __name__ == "__main__":
    main()