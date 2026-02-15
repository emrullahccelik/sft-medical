import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from app.config import CACHE_DIR, SAVE_DIR


def generate_response(prompt):

    # Ayarlar
    base_model_id = "Qwen/Qwen3-0.6B"
    adapter_dir = f"{SAVE_DIR}{base_model_id}/"
    
    print(f"Base Model: {base_model_id}")
    print(f"Adapter Path: {adapter_dir}")

    # 1. Base Modeli Yükle
    print("Base model yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=False,
        cache_dir = CACHE_DIR
    )

    # 2. LoRA Adapter'ı Yükle
    print("LoRA adapter yükleniyor...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    # 3. Prompt Hazırla
    system_prompt = "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın. Soruları doğru, kapsamlı ve anlaşılır biçimde yanıtla."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 4. Generate
    # we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
    print("Yanıt üretiliyor...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
    )
    
    # 5. Çıktıyı Formatla
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

if __name__ == "__main__":
    test_prompt = "Hamilelikte baş ağrısı için hangi ilaçlar güvenlidir?"
    print(f"\nSoru: {test_prompt}\n")
    response = generate_response(test_prompt)
    print(f"Cevap: {response}")
