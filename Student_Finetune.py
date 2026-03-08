# @title Student_finetune.py

#!pip install datasets
#!pip install trl
#!pip install unsloth

import json
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

#Config Setings:
STUDENT_MODEL_ID = "unsloth/granite-4.0-350m"
TEACHER_RESPONSES_FILE = "teacher_responses.json"
OUTPUT_DIR = "./student_lora_adapter"
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 2
NUM_EPOCHS = 3

LORA_RANK          = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
TARGET_MODULES     = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

PER_DEVICE_TRAIN_BATCH  = 2
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE           = 2e-4
WARMUP_RATIO            = 0.05
LR_SCHEDULER            = "cosine"
WEIGHT_DECAY            = 0.01
LOGGING_STEPS           = 10
SAVE_STRATEGY           = "epoch"
FP16                    = False
BF16                    = False


def load_teacher_responses(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_conversation(item: dict) -> dict:
    messages = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    return {"messages": messages}


if __name__ == "__main__":
    print("Loading teacher responses...")
    raw_data = load_teacher_responses(TEACHER_RESPONSES_FILE)
    formatted = [format_conversation(item) for item in raw_data]
    dataset = Dataset.from_list(formatted)
    print(f"Dataset size: {len(dataset)} examples\n")

    print(f"Loading student model: {STUDENT_MODEL_ID}")
    model, tokenizer = FastModel.from_pretrained(
        model_name      = STUDENT_MODEL_ID,
        max_seq_length  = MAX_SEQ_LENGTH,
        load_in_4bit    = True,
    )

    model = FastModel.get_peft_model(
        model,
        #finetune_vision_layers=False,
        #finetune_language_layers=True,
        #finetune_attention_modules=True,
        #finetune_mlp_modules=True,
        r=LORA_RANK,
        target_modules      = TARGET_MODULES,
        lora_alpha          = LORA_ALPHA,
        lora_dropout        = LORA_DROPOUT,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",
        random_state        = 42,
        use_rslora          = False,
        loftq_config        = None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    print("Student model loaded with LoRA adapters.\n")

    def apply_template(batch):
        batch["text"] = tokenizer.apply_chat_template(
            batch["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return batch

    dataset = dataset.map(apply_template, batched=True)

    sft_config = SFTConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH,
        gradient_accumulation_steps = GRAD_ACCUMULATION_STEPS,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = LR_SCHEDULER,
        weight_decay                = WEIGHT_DECAY,
        logging_steps               = LOGGING_STEPS,
        save_strategy               = SAVE_STRATEGY,
        fp16                        = FP16,
        bf16                        = BF16,
        optim                       = "adamw_8bit",
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
        packing                     = True,
        report_to                   = "none",
        seed                        = 42,
    )

    trainer = SFTTrainer(
        model      = model,
        tokenizer  = tokenizer,
        train_dataset = dataset,
        args       = sft_config,
    )

    print("Starting fine-tuning...")
    torch.cuda.empty_cache()
    trainer.train()
    torch.cuda.empty_cache()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nStudent LoRA adapter saved to {OUTPUT_DIR}")