import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from app.config import CACHE_DIR, SAVE_DIR
from app.core.logger import logger

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın. Soruları doğru, kapsamlı ve anlaşılır biçimde yanıtla."
MAX_SEQ_LENGTH = 1024


def load_model():
    """Fine-tuned modeli ve tokenizer'ı yükle"""
    adapter_path = f"{SAVE_DIR}{MODEL_ID}/"
    logger.info(f"Model yükleniyor: {MODEL_ID}")
    logger.info(f"Adapter path: {adapter_path}")

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


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Verilen prompt için yanıt üret"""
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Sadece üretilen kısmı al
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


if __name__ == "__main__":
    model, tokenizer = load_model()

    test_prompts = [
        "Hamilelikte baş ağrısı için hangi ilaçlar güvenlidir?",
        "Hipertansiyon nedir ve tedavisi nasıl yapılır?",
        "Tip 2 diyabet hastalarında beslenme önerileri nelerdir?",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Soru: {prompt}")
        print(f"{'='*60}")
        response = generate_response(model, tokenizer, prompt)
        print(f"Cevap: {response}")
