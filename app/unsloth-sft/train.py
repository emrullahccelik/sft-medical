import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from app.core.logger import logger
from app.config import CACHE_DIR, OUTPUT_DIR, SAVE_DIR

# NOT: Tıbbi metinler uzun olabilir. Veri kesilme riskini dataset'teki
# ortalama token uzunluğunu kontrol ederek değerlendir.
max_seq_length = 1024  # Küçük GPU için azaltıldı
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
model_id = "unsloth/Llama-3.2-1B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    cache_dir=CACHE_DIR,
    trust_remote_code=False,
)


model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,  # 2*r oranı daha stabil öğrenme sağlar
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # VRAM tasarrufu için
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


SYSTEM_PROMPT = "Sen tıp alanında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka asistanısın. Soruları doğru, kapsamlı ve anlaşılır biçimde yanıtla."
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)


def formatting_prompts_func(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return {"text": texts}


dataset = load_dataset("turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1")
logger.info(f"Dataset: {dataset}")

dataset = dataset.map(formatting_prompts_func, batched=True)
logger.info(f"Dataset: {dataset}")

print(dataset["train"][0])

num_train_epochs = 1
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 4  # Effective batch = 2x4 = 8
logging_steps = 10
save_strategy = "steps"
save_steps = 1000  # ~5900 step/epoch, epoch boyunca ~5 kez kaydeder
eval_strategy = "steps"
eval_steps = 1000  # save_steps ile uyumlu

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,  # Kısa sekanslar için 5x hızlandırma (kendi collator'ını kullanır)
    args=TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,  # VRAM tasarrufu için eval de 1'er 1'er
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"{OUTPUT_DIR}{model_id}/",
        report_to="tensorboard",  # Use this for WandB etc
    ),
)


trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)


print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


save_path = f"{SAVE_DIR}{model_id}/"
logger.info(f"Model kaydediliyor: {save_path}")
model.save_pretrained(save_path)  # Local saving
tokenizer.save_pretrained(save_path)
